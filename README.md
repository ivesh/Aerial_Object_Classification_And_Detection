# Aerial_Object_Classification_And_Detection
This project aims to develop a computer vision based solution that can classify aerial images into two categories — Bird or Drone. It will perform object detection and classification to locate and label these objects in real-world scenes.

The solution will help in security surveillance, wildlife protection, and airspace safety where accurate identification between drones and birds is critical. The project involves building a Custom CNN classification model, leveraging transfer learning, and implementing YOLOv8 for real-time object detection. The final solution will be deployed using Streamlit for interactive use.

Important tools: 
Roboflow: https://app.roboflow.com/computervision-pfnmk


#Project Structure:
1. Data Collection: Data can be collected in 3 ways
    - Collect the ready made data available in the internet - (In this project we have already collected data which is annotated as well using roboflow)
    - Have a team of data annotators and send them to the field collect and annotate data
    - Web Scraping script to collect necessary images and annotate the images
2. Data Annotation: We can either annotate the data using labelImg or roboflow.  
    - Annotated the data using roboflow object detection. During data annotation step, we have to provide the images to annotate and in the next page options will be there to create version of the data where we can perform train test split using the images provided, preprocessing steps such as Auto Orient and Resize is available, and data augmentation. 
    - After the version of this data is ready we can train with the model using roboflow(paid feature), however we will be training this images here inside our project workflow.
    - Post theversion of the data is ready we can download it which will contain the images, labels and data.yaml(contains path of the image and number of classes) files.
3. Project Workflow: 
    - 1st Step: Updated custom logging under AerialObjectDetectionAndClassificaation->logger->__init__.py & updated exception under AerialObjectDetectionAndClassificaation->exception->__init__.py
    - 
    - Components: Main components of data ingestion, data validation, model pusher and model trainer components
