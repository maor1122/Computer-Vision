# Computer-Vision
<h3>
Some of the projects in the course "Computer Vision" - which was on the 2nd year in my CS degree.
Plus one project made after, "Mask Detection".

showcase of each project:
</h3>

# Augmented reality
<h3>
  In this project, we want to switch an item in a given video
  first let's see the inputted video:
  <br>
  <br><img src="res/inputVid.gif" width="300" height="500"/><br>
  We want to switch the image in the video with this photo:
  <br>
  <br><img src="Augmented reality/newImage.jpg" width="200" height="200"/><br>
  
  The result:
  <br>
  <br><img src="res/outputVid.gif" width="300" height="500"/><br>
 you can try it yourself! switch the file in path "Augmented reality/newImage.jpg" with a new image and run "Augmented reality/perspective_warping.py".
 
 # Lane Detection
 <h3>
  in this project we'll detect lanes and lane changes in a given video.
  for example with this video:
  <br>
  <br><img src="res/roadtrip.gif" width="500" height="300"/><br>
  
  after running the program we'll have this video:
  <br>
  <br><img src="res/roadtripOutput.gif" width="600" height="300"/><br>
 you can try it yourself! switch the file in path "Lane detection/roadtrip.mp4" with a new video and run "Lane detection/projectLanes.py".
 </h3>

 # mask detection

 In this project I showcased how I trained a model with very limited data, <b>only 100 images</b>
 I used YOLO v8 model which already trained on the COCO dataset, and since the data is a little similar - training in on mask detection should be possible even with that amount of data.
 The way we do that is freeze the first layers of the model. 
 The first layers are responsible for more abstract data patterns (for example edges), then we need to train the deeper layers that are responisble for more complex data pattarns (for example cloths).
 Another importent thing is to use augmentation when training the model, since we have very limited data.
Using augmentation will make the model think there are many images instead of the only 100 we provided. It works by flipping, zooming in and changing colors of the images we provided and by doing that, it creates more data to learn from.
Note that having more data is obviously better and will train a better model, this project is just a showcase of how to deal with less data.
example result:
<br><img width="772" height="517" alt="image" src="https://github.com/user-attachments/assets/087fd650-168a-4d9e-adbb-0b10c0767ca0" /><br>

