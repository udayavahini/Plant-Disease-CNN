# Plant & Urban Tree Disease Detection with CNN

This project uses Convolutional Neural Networks (CNNs) to identify diseases in plant leaves and urban trees from images. 
It covers data preprocessing, model design, and a user-facing interface. This is an intial version limited to data set containing Tomato, potato and Bell Pepper Crops. Will be expanded to further training in future


## Structure
- `data/`: Dataset imported using kaggle API 
- `.kaggle`: Kaggle.json to import and save dataset
- `notebooks/`: Jupyter notebooks for exploration/training
- `app/`: Streamlit app for interactive demos
- `deployment/`: Deployment on Streamlit cloud server

## Data Preprocessing
- Splitting Dataset into Train and Test 
- Image Compression and augmentation
- Visualize class balance, sample images.

## Modelling Steps

Summarized modeling in notebooks and as a script in `src/`

## App creation

Local app creation and designing done in `app/streamlit.py`

## App Deployment 

App deployment done on streamlit server using folder `deployment` (Link for web app : https://plantdoctor-uv.streamlit.app/)

## Git Commands to run the app

cd app
streamlit run streamlit.py

## References

- [Kaggle Datasets and API](https://www.kaggle.com/docs/api)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [GitHub Docs](https://docs.github.com/en)

---

END OF README