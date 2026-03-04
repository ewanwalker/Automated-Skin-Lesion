Running the API then calling it with an image file.

Step 1:
Starting the server

cd your-file-path/Automated-Skin-Lesion

python main.py


Step 2:
Calling the api

curl -X POST http://localhost:8000/images \ -F "file=@path-to-image"
