# Human Activity Recognition 
The aim of this project is to automatically detect Human Activities based on data generated from a mobile sensory device, using machine learning. The machin learning model was trained on accelerometer and gyroscope data generated while some individuals were carrying out these activities;
* Eating Crunchy Food
* Drinking Water
* Speaking 
* Eating soft food 

![image](https://user-images.githubusercontent.com/38056084/138591749-ddd5ed0e-8904-4c54-a53e-6ba931f44d00.png)

Fig. 1 Cruncy Food [Source](https://www.google.com/imgres?imgurl=https%3A%2F%2Fs3.amazonaws.com%2Fsecretsaucefiles%2Fphotos%2Fimages%2F000%2F103%2F517%2Flarge%2FBowl-of-Cereal.jpg%3F1485310468&imgrefurl=https%3A%2F%2Fspoonuniversity.com%2Flifestyle%2Fsoggy-food-worse-than-crunchy-food&tbnid=FtttNQqWb30fBM&vet=12ahUKEwiX_PKp9uLzAhUa0YUKHU4sC58QMygAegUIARDKAQ..i&docid=KLhVsvC4NRS5pM&w=770&h=513&q=crunchy%20food&ved=2ahUKEwiX_PKp9uLzAhUa0YUKHU4sC58QMygAegUIARDKAQ)

![image](https://user-images.githubusercontent.com/38056084/138591843-be5e8c19-e428-4017-8b77-49e9396f95f5.png) 

Fig.2 Drinking Water [Source](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.healthline.com%2Fhealth%2Ffood-nutrition%2Fwhy-is-water-important&psig=AOvVaw1Qidozcl8C_auAnHLYOp8A&ust=1635160816744000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCIC2l8Dp4vMCFQAAAAAdAAAAABAD)

![image](https://user-images.githubusercontent.com/38056084/138591964-0b9cc682-0674-44fa-84a2-57d572dfaad7.png) 

Fig.3 Speaking [Source](https://www.google.com/url?sa=i&url=https%3A%2F%2Feducationalresearchtechniques.com%2F2017%2F07%2F21%2Ftypes-of-speaking-in-esl%2F&psig=AOvVaw3k0gpJg47Nree7h3OH5JN8&ust=1635161001932000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCJDOq-bq4vMCFQAAAAAdAAAAABAD)


![image](https://user-images.githubusercontent.com/38056084/138592110-55e1cfdf-b3f0-424c-843e-b17c2914286f.png)

Fig.4 Soft Food [Source](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.eatthis.com%2Fsoft-foods-diet%2F&psig=AOvVaw2qndGsXoyFHYiUntqCqz_q&ust=1635161224256000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCIiluv3q4vMCFQAAAAAdAAAAABAJ)

## Dataset 
Sensory Device data: The Dataset consists of 4 types of Activities which were collected from 6 persons.

## Dataset Description
![image](https://user-images.githubusercontent.com/38056084/138592283-b5c21afa-625f-4034-9ccf-1f2cb58a1977.png)

## Random Forest Model Architecture 
```py
rf_new = RandomForestClassifier(**{'n_estimators': 1400,
 'min_samples_split': 2,
 'min_samples_leaf': 1,
 'max_features': 'auto',
 'max_depth': 40,
 'bootstrap': False})
 
 rf_new.fit(X_train, y_train)
```
* Random Forest Confusion Matrix on the Dataset
![image](https://user-images.githubusercontent.com/38056084/138592415-239495c8-172c-480e-bfaf-a577170358b2.png)

* Random Forest Classification Report on the Dataset
![image](https://user-images.githubusercontent.com/38056084/138592457-e9aa2303-902c-47e5-a89f-c79790bb9828.png)




