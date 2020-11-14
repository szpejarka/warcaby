import * as tf from '@tensorflow/tfjs-node';
import { range } from '@tensorflow/tfjs-node';
 
// Define a model for linear regression.

var model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
model = await tf.loadLayersModel('file://my-model1/model.json');
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
// Train the model using the data.
await model.fit(xs, ys, {epochs: 10}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([5], [1, 1])).print();
  // Open the browser devtools to see the output
});
await model.save('file://my-model1');
console.log("ok");
