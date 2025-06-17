import cv2
import joblib
import numpy as np
import sys
import torch
from PIL import Image
import os
from insightface.app import FaceAnalysis
import psycopg2

# Import LPRNet and MTCNN modules
sys.path.append('./LPRNet')
sys.path.append('./MTCNN')
from LPRNet_Test import LPRNet,STNet, CHARS, decode
from MTCNN import create_mtcnn_net
import classifier

def check_complete_match(car_model, plate_number, person_name):
    # Correct the connection string format
    conn = psycopg2.connect("dbname='access_qu_database' user='sdp2' password='2000' host='localhost'")
    cursor = conn.cursor()
    
    # Adjusted query to handle different matching scenarios
    query = """
    SELECT c.Car_Model, p.Name, lp.License_Plate_Number
    FROM Car_Ownership co
    JOIN Cars c ON co.Car_ID = c.Car_ID
    JOIN People p ON co.Person_ID = p.Person_ID
    JOIN License_Plates lp ON co.Plate_ID = lp.Plate_ID
    WHERE (
        (c.Car_Model = %s AND lp.License_Plate_Number = %s AND p.Name = %s) OR
        (c.Car_Model = %s AND lp.License_Plate_Number = %s) OR
        (lp.License_Plate_Number = %s AND p.Name = %s)
    );
    """
    cursor.execute(query, (car_model, plate_number, person_name, car_model, plate_number, plate_number, person_name))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result is not None  # Returns True if a matching record is found, otherwise False



def bbox_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    inter_area = max(inter_rect_x2 - inter_rect_x1, 0) * max(inter_rect_y2 - inter_rect_y1, 0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / float(b1_area + b2_area - inter_area)
    return iou

# Load YOLO model
net = cv2.dnn.readNet("yolo-coco/yolov4-tiny.weights", "yolo-coco/yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
classes = []
with open("yolo-coco/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

cap = cv2.VideoCapture("testvid4.MOV")
min_confidence = 0.3
iou_threshold = 0.41
best_frame = None
frame_skip = 60  # Process every 20th frame
frame_count = 0

while True:
    ret, original_frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip this frame

    # Resize frame for processing
    frame = cv2.resize(original_frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    height, width = frame.shape[:2]

    # Define the reference bounding box (centered and with predefined size)
    box_size = (width // 2, height // 2)  # Size of the IoU reference box
    offset_x, offset_y = -20, 75 # Adjust these to move the box around
    """"""""""
    Right: Increase offset_x
    Left: Decrease offset_x
    Up: Decrease offset_y
    Down: Increase offset_y
    """""""""
    ref_box = [
        (width // 2) - (box_size[0] // 2) + offset_x,
        (height // 2) - (box_size[1] // 2) + offset_y,
        (width // 2) + (box_size[0] // 2) + offset_x,
        (height // 2) + (box_size[1] // 2) + offset_y
    ]

    # Draw the IoU reference box
    #cv2.rectangle(frame, (ref_box[0], ref_box[1]), (ref_box[2], ref_box[3]), (0, 255, 0), 2)


    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence:
                center_x, center_y, w, h = detection[0:4] * np.array([width, height, width, height])
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                detected_box = [x, y, x + int(w), y + int(h)]
                iou = bbox_iou(detected_box, ref_box)
                if iou >= iou_threshold:
                    best_frame = frame
                    break
        if best_frame is not None:
            break

    if best_frame is not None:
        break

cap.release()

if best_frame is not None:
    # Convert the best frame to the desired format for processing
    #cv2.imshow("Best Frame with IoU Box", best_frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # Convert the best frame to the desired format for processing
    best_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    best_frame_image = np.array(best_frame)
else:
    print("No suitable frame found based on IoU threshold.")

# Further code for car color classification and other tasks follows here...

    exit()

# Assuming the best_frame_image is ready for the second part
car_color_classifier = classifier.Classifier()
LABELS = classes  # Reuse the loaded classes

# Since we're not using command-line arguments, set configurations directly
confidence_setting = 0.5
threshold_setting = 0.3

image = best_frame_image  # Use the best frame image directly
(H, W) = image.shape[:2]

# ... (Yolo configuration and loading as before)

# Forward pass of YOLO to get bounding boxes and probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)

boxes = []
confidences = []
classIDs = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > confidence_setting:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_setting, threshold_setting)

# Assuming you have lists or dictionaries to store each detection
detected_cars = []
detected_plates = []
detected_faces = []

# Loop through each detection for classification
if len(idxs) > 0:
    for i in idxs.flatten():
        x, y, w, h = boxes[i]
        color = [int(c) for c in np.random.randint(100, 255, size=(3,))]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 206, 209), 2)
        if classes[classIDs[i]] == "car":
            car_region = image[max(y, 0):y + h, max(x, 0):x + w]
            results = car_color_classifier.predict(car_region)
            for result in results:
                if float(result['prob']) > 0.4:  # Check if the probability is greater than 0.4
                    make_model = "{} {}".format(result['make'], result['model'])
                    car_info = { 'Car_model': make_model, 'confidence': result['prob'] }
                    detected_cars.append(car_info)

                    #label_text = "{} {} {:.4f}".format(result['make'], result['model'], float(result['prob']))
                    #cv2.putText(image, label_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# cv2.imwrite("classified_car.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Optionally display the image or save it to disk

# cv2.imshow("Classified Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


## Thamer
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0.5, lpr_max_len=7, phase='test').to(device)
lprnet.load_state_dict(torch.load('LPRNet/weights/good.pth', map_location=lambda storage, loc: storage))
lprnet.eval()
STN = STNet().to(device)
STN.load_state_dict(torch.load('LPRNet/weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
STN.eval()

# Replace 'best_frame_image' with the actual variable holding the best frame selected for car detection
image = cv2.cvtColor(np.array(best_frame_image), cv2.COLOR_RGB2BGR)  # Ensure the image is in BGR format for OpenCV
scale = 1  # Scale if necessary
mini_lp = (50, 15)  # Minimum size of the license plate
image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
bboxes = create_mtcnn_net(image, mini_lp, device, p_model_path='MTCNN/weights/pnet_Weights',
                          o_model_path='MTCNN/weights/onet_Weights')

for i in range(bboxes.shape[0]):
    bbox = bboxes[i, :4]
    x1, y1, x2, y2 = [int(bbox[j]) for j in range(4)]
    w = int(x2 - x1 + 1.0)
    h = int(y2 - y1 + 1.0)
    img_box = image[y1:y2 + 1, x1:x2 + 1, :]
    im = cv2.resize(img_box, (94, 24), interpolation=cv2.INTER_CUBIC)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
    data = torch.from_numpy(im).float().unsqueeze(0).to(device)
    transfer = STN(data)
    #preds = lprnet(transfer)
    preds = lprnet(data)
    preds = preds.cpu().detach().numpy()

    labels, pred_labels, confidences = decode(preds, CHARS)

    # Drawing bounding box and displaying text
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 204, 255), 2)

    for label, confidence in zip(labels, confidences):
        confidence_text = f"{label} {np.mean(confidence):.2f}"  # Average confidence for the label
        plate_info = {'plate': label, 'confidence': np.mean(confidence) }

        detected_plates.append(plate_info)
        # Display the label with its confidence score right below the bounding box
        #text_position = (x1, y1 - 10)
        #cv2.putText(image, confidence_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Load the trained face recognition model and label encoder
classifier = joblib.load('face_recognizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize and prepare the face analysis application
app = FaceAnalysis()
app.prepare(ctx_id=0)

# Detect Image

faces = app.get(image)

# Recognize faces
for face in faces:
    try:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        face_embedding = face.normed_embedding
        pred_label_num = classifier.predict([face_embedding])[0]
        pred_label = label_encoder.inverse_transform([pred_label_num])[0]

        # Get the confidence score
        probabilities = classifier.predict_proba([face_embedding])[0]
        confidence = max(probabilities)  # Get the highest probability
        confidence_percent = f"{confidence:.2f}"  # Convert to percentage

        face_info = {'name': pred_label, 'confidence': confidence}
        detected_faces.append(face_info)

        # Draw bounding box and write the label and confidence
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        #cv2.putText(image, f"{pred_label} ({confidence_percent})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
         #           (0, 255, 0), 2)
    except Exception as e:
        print(f"Error in recognition: {e}")



# Now you have lists of detected items with their details
#print(detected_cars)
#print(detected_plates)
#print(detected_faces)

# Assuming all detections are stored and you have example detections
example_detected_car = detected_cars[0] if detected_cars else None
example_detected_plate = detected_plates[0] if detected_plates else None
example_detected_person = detected_faces[0] if detected_faces else None

print(example_detected_car)
print(example_detected_plate)
print(example_detected_person)
# Define colors
green = (0, 230, 0)  # BGR format for green
red = (0, 0, 255)    # BGR format for red
black = (0, 0, 0)    # BGR format for black
# Define default values for names and confidence scores
person_name = 'Unknown'
car_model = 'Unknown'
plate_number = 'Unknown'
face_confidence = 0.0
car_confidence = 0.0
plate_confidence = 0.0


def classify_access(car_confidence, plate_confidence, face_confidence):
    # Provide default values for confidences if any data is missing
    car_confidence = float(car_confidence) if car_confidence else 0.0
    plate_confidence = float(plate_confidence) if plate_confidence else 0.0
    face_confidence = float(face_confidence) if face_confidence else 0.0
    score = (2.103 * car_confidence) + (6.27 * plate_confidence) + (3.392 * face_confidence) - 7.588
    return score

# Use detected data to get confidences, default to '0.0' if not present
car_confidence = float(example_detected_car['confidence']) if example_detected_car and 'confidence' in example_detected_car else 0.0
plate_confidence = float(example_detected_plate['confidence']) if example_detected_plate and 'confidence' in example_detected_plate else 0.0
face_confidence = float(example_detected_person['confidence']) if example_detected_person and 'confidence' in example_detected_person else 0.0

# If any detection is available, proceed to check
if example_detected_car or example_detected_plate or example_detected_person:
    # Provide None for missing data to match optional SQL checks
    car_model = example_detected_car['Car_model'] if example_detected_car else 'Unknown'
    plate_number = example_detected_plate['plate'] if example_detected_plate else 'Unknown'
    person_name = example_detected_person['name'] if example_detected_person else 'Unknown'
    
    # Check matches with the database
    if check_complete_match(car_model, plate_number, person_name):
        # Calculate the classification score
        score = classify_access(car_confidence, plate_confidence, face_confidence)
        
        if score > 0:
            # Draw a black rectangle as background for text
            cv2.rectangle(image, (5, 5), (360, 130), black, -1)
            cv2.putText(image, "Match found - Access Granted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)
            cv2.putText(image, f"{person_name}: {100*face_confidence: .2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"{car_model}: {100*car_confidence: .2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (209,206,0), 2)
            cv2.putText(image, f"{plate_number}: {100*plate_confidence: .2f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 204, 255), 2)
            print("Match found - Access Granted")
            print(score)
            # Here you could trigger an action like opening a gate
        else:
            cv2.rectangle(image, (5, 5), (350, 130), black, -1)
            cv2.putText(image, "Match found - Must verify!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,102,255), 2)
            cv2.putText(image, f"{person_name}: {100*face_confidence: .2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"{car_model}: {100*car_confidence: .2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (209,206,0), 2)
            cv2.putText(image, f"{plate_number}: {100*plate_confidence: .2f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 204, 255), 2)
            print("Match found with low confidence - Check Manually")
            print(score)
    else:
        cv2.rectangle(image, (5, 5), (350, 130), black, -1)
        cv2.putText(image, "No match found - Access Denied", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
        cv2.putText(image, f"{person_name}: {100*face_confidence: .2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"{car_model}: {100*car_confidence: .2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (209,206,0), 2)
        cv2.putText(image, f"{plate_number}: {100*plate_confidence: .2f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 204, 255), 2)
        print("No match found - Access Denied")
else:
    cv2.rectangle(image, (5, 5), (350, 130), black, -1)
    cv2.putText(image, "No enough data", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
    cv2.putText(image, f"{person_name}: {100*face_confidence: .2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"{car_model}: {100*car_confidence: .2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (209,206,0), 2)
    cv2.putText(image, f"{plate_number}: {100*plate_confidence: .2f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 204, 255), 2)
    print("Not enough data to perform matching.")






cv2.imwrite("final_result.jpg", image)
# Optionally display the image or save it to disk
cv2.imshow("Result Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
