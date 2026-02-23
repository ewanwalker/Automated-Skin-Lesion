Running the API then calling it with an image file.

Step 1:
Starting the server

cd your-file-path/Automated-Skin-Lesion

uvicorn main:app --reload


Step 2:
Calling the api

cd your-file-path/Automated-Skin-Lesion/dataset/val/benign

curl -X POST http://localhost:8000/images \ -F "file=@ISIC_0013140.jpg"
