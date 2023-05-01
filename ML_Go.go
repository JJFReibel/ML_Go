package main
import (
"math"
"math/rand"
)
// Go ML
// By JJ Reibel

func TrainValTestSplit(X []float64, y []float64, valSize float64, testSize float64, epochs int, randomState int64) ([][]float64, [][]float64, [][]float64, [][]float64, [][]float64, [][]float64) {
// Get the total number of samples in the dataset
nSamples := len(X)
// Set the random seed if provided
if randomState != 0 {
rand.Seed(randomState)
}
// Create a list of indices that correspond to the samples in the dataset
idx := make([]int, nSamples)
for i := range idx {
idx[i] = i
}
// Shuffle the indices
rand.Shuffle(nSamples, func(i, j int) {
idx[i], idx[j] = idx[j], idx[i]
})
// Calculate the number of samples to allocate to the validation and test sets
nVal := int(math.Ceil(float64(nSamples) * valSize))
nTest := int(math.Ceil(float64(nSamples) * testSize))
// Initialize the starting and ending indices of each epoch
epochStartIdx := make([]int, epochs)
for i := range epochStartIdx {
epochStartIdx[i] = i * nSamples / epochs
}
epochEndIdx := make([]int, epochs)
copy(epochEndIdx, epochStartIdx[1:])
epochEndIdx[len(epochEndIdx)-1] = nSamples
// Initialize the slices to hold the indices of the samples in each set for each epoch
trainIdxEpoch := make([][]int, epochs)
valIdxEpoch := make([][]int, epochs)
testIdxEpoch := make([][]int, epochs)
// Loop through each epoch
for i := 0; i < epochs; i++ {
// Get the indices of the samples in the current epoch
epochIndices := idx[epochStartIdx[i]:epochEndIdx[i]]
// Calculate the indices of the samples to allocate to the validation and test sets
valIdx := epochIndices[:nVal]
testIdx := epochIndices[nVal : nVal+nTest]
trainIdx := epochIndices[nVal+nTest:]
// Add the indices to the appropriate slices for the current epoch
trainIdxEpoch[i] = trainIdx
valIdxEpoch[i] = valIdx
testIdxEpoch[i] = testIdx
}
// Initialize slices to hold the data for each epoch
XTrainEpoch := make([][]float64, epochs)
XValEpoch := make([][]float64, epochs)
XTestEpoch := make([][]float64, epochs)
yTrainEpoch := make([][]float64, epochs)
yValEpoch := make([][]float64, epochs)
yTestEpoch := make([][]float64, epochs)
// Loop through each epoch
for i := 0; i < epochs; i++ {
// Get the indices of the samples for the current epoch
trainIdx := trainIdxEpoch[i]
valIdx := valIdxEpoch[i]
testIdx := testIdxEpoch[i]
// Get the data for the current epoch
XTrain := make([]float64, len(trainIdx))
XVal := make([]float64, len(valIdx))
XTest := make([]float64, len(testIdx))
yTrain := make([]float64, len(trainIdx))
yVal := make([]float64, len(valIdx))
yTest := make([]float64, len(testIdx))
for i := range trainIdx {
XTrain[i] = X[trainIdx[i]]
yTrain[i] = y[trainIdx[i]]
}
for i := range valIdx {
XVal[i] = X[valIdx[i]]
yVal[i] = y[valIdx[i]]
}
for i := range testIdx {
XTest[i] = X[testIdx[i]]
yTest[i] = y[testIdx[i]]
}
// Append the data to the appropriate lists for the current epoch
XTrainEpoch = append(XTrainEpoch, XTrain)
XValEpoch = append(XValEpoch, XVal)
XTestEpoch = append(XTestEpoch, XTest)
yTrainEpoch = append(yTrainEpoch, yTrain)
yValEpoch = append(yValEpoch, yVal)
yTestEpoch = append(yTestEpoch, yTest)
}
// Return the data for each epoch as six slices
return XTrainEpoch, XValEpoch, XTestEpoch, yTrainEpoch, yValEpoch, yTestEpoch
}


/ Example
// XTrainEpoch, XValEpoch, XTestEpoch, yTrainEpoch, yValEpoch, yTestEpoch := trainValTestSplit(X, y, 0.1, 0.1, 5)
// Can loop through
// XTrainEpoch[0] // training data for epoch 0
// XValEpoch[0] // validation data for epoch 0
// XTestEpoch[0] // test data for epoch 0
// yTrainEpoch[0] // training labels for epoch 0
// yValEpoch[0] // validation labels for epoch 0
// yTestEpoch[0] // test labels for epoch 0
