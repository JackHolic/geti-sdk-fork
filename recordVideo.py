import cv2
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'C:\Users\dangkho1\OneDrive - Intel Corporation\Documents\MAE\AI\Geti\Video\myvid.mp4', fourcc, 20.0, (640,480))
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

print("Press 'q' to stop recording.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Write the frame into the output file
    out.write(frame)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Press 'q' on the keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
