import numpy as np

import matplotlib.pyplot as plt

# Import `train_test_split`
from sklearn import datasets
from sklearn import svm
from sklearn import cluster
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import scale
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.grid_search import GridSearchCV

# Load in the data
# Load in the `digits` data
digits = datasets.load_digits()

# # # # # # # # # # # # # # # # # # # # # # # # # # 
# Plot some pictures 
## # # # # # # # # # # # # # # # # # # # # # # # # 
# Figure size (width, height) in inches
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

# Show the plot
plt.savefig('numbersToLearn.png')
plt.close() 

# # # # # # # # # # # # # # # # # # # # # # # # # #
# Look at PCA
# # # # # # # # # # # # # # # # # # # # # # # # # #
# Create a Randomized PCA model that takes two components
randomized_pca = RandomizedPCA(n_components=2)

# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(digits.data)

# Create a regular PCA model 
pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(digits.data)

# Inspect the shape
reduced_data_pca.shape

# Print out the data
# print(reduced_data_rpca) 
# print(reduced_data_pca)
fig = plt.figure(figsize=(8, 6))
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    x = reduced_data_rpca[:, 0][digits.target == i]
    y = reduced_data_rpca[:, 1][digits.target == i]
    plt.scatter(x, y, c=colors[i])
plt.legend(digits.target_names)#, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.savefig('rPCAScatterPlot.png')
plt.close()

# # # # # # # # # # # # # # # # # # # # # # # # # #
# Setup the Data to be fitted
# # # # # # # # # # # # # # # # # # # # # # # # # #

# The first thing that we’re going to do is preprocessing the data. You can standardize the digits data by, for example, making use of the scale() method:
data = scale(digits.data)

# Split the `digits` data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Actually fit the data
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# # Create the KMeans model
# clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

# # Fit the training data to the model
# clf.fit(X_train)

# # Figure size in inches
# fig = plt.figure(figsize=(8, 3))

# # Add title
# fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# # For all labels (0-9)
# for i in range(10): 
#     # Initialize subplots in a grid of 2X5, at i+1th position
#     ax = fig.add_subplot(2, 5, 1 + i)
#     # Display images
#     ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
#     # Don't show the axes
#     plt.axis('off')

# # Show the plot
# plt.savefig('clusterCentres.png')
# plt.close()


# # Predict the labels for `X_test`
# y_pred=clf.predict(X_test)

# # Print out the first 100 instances of `y_pred`
# # Print out the first 100 instances of `y_test`
# for i in range(10):
# 	print("test = " + str(y_test[i]) + " , pred =" + str(y_pred[i]))

# # Create an isomap and fit the `digits` data to it
# X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# # Compute cluster centers and predict cluster index for each sample
# clusters = clf.fit_predict(X_train)

# # Create a plot with subplots in a grid of 1X2
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# # Adjust layout
# fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
# fig.subplots_adjust(top=0.85)

# # Add scatterplots to the subplots 
# ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
# ax[0].set_title('Predicted Training Labels')
# ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
# ax[1].set_title('Actual Training Labels')
# plt.savefig('clusterCentres.png')
# plt.close("PredictedTrainingLabels.png")

# print("")
# print("Confusion Matrix:")
# print(metrics.confusion_matrix(y_test, y_pred))

# print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
# print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
#           %(clf.inertia_,
#       homogeneity_score(y_test, y_pred),
#       completeness_score(y_test, y_pred),
#       v_measure_score(y_test, y_pred),
#       adjusted_rand_score(y_test, y_pred),
#       adjusted_mutual_info_score(y_test, y_pred),
#       silhouette_score(X_test, y_pred, metric='euclidean')))

# # # # # # # # # # # # # # # # # # # # # # # # # #
# Fit the data with a different moddel 
# # # # # # # # # # # # # # # # # # # # # # # # # #

print("")
print("SVC Model")
# How to find good paramaters for the SVC Model

# Set the parameter candidates
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# Create a classifier with the parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on training data
clf.fit(X_train, y_train)

# Print out the results 
print('Best score for training data:', clf.best_score_)
print('Best `C`:',clf.best_estimator_.C) #10
print('Best kernel:',clf.best_estimator_.kernel) #rfb
print('Best `gamma`:',clf.best_estimator_.gamma) #gamma

# Apply the classifier to the test data, and view the accuracy score
clf.score(X_test, y_test)  

# Train and score a new classifier with the grid search parameters
svc_model = svm.SVC(C=10, kernel='rbf', gamma=0.001).fit(X_train, y_train)

svc_model.fit(X_train, y_train)

# Import matplotlib
import matplotlib.pyplot as plt

# Assign the predicted values to `predicted`
predicted = svc_model.predict(X_test)

# Zip together the `images_test` and `predicted` values in `images_and_predictions`
images_and_predictions = list(zip(images_test, predicted))

# For the first 4 elements in `images_and_predictions`
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    # Initialize subplots in a grid of 1 by 4 at positions i+1
    plt.subplot(1, 4, index + 1)
    # Don't show axes
    plt.axis('off')
    # Display images in all subplots in the grid
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    # Add a title to the plot
    plt.title('Predicted: ' + str(prediction))

# Show the plot
plt.savefig('first4ElementsIn_images_and_predictions.png')
plt.close()

# Print the classification report of `y_test` and `predicted`
print(metrics.classification_report(y_test, predicted))

# Print the confusion matrix
print(metrics.confusion_matrix(y_test, predicted))

# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
predicted = svc_model.predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust the layout
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots 
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=predicted)
ax[0].set_title('Predicted labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Labels')


# Add title
fig.suptitle('Predicted versus actual labels', fontsize=14, fontweight='bold')

# Show the plot
plt.savefig('PredictedVersusActualLabels')
plt.close()