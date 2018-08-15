
const tf = require("@tensorflow/tfjs");
require('@tensorflow/tfjs-node');
tf.setBackend('tensorflow');

const gaussian = require('gaussian');

module.exports = class EM {
  constructor(data, classes) {
    this.data = tf.tensor(data);
    let sigma = tf.sum(this.data.sub(this.data.mean(0)).pow(2)).div(data.length);
    this.gaus1 = {
        mu: this.data.mean(0).add(Math.random()*100 + 10),
        sigma: sigma,
        prob: 0.5
    }
    this.gaus2 = {
        mu: this.data.mean(0).sub(Math.random()*100 + 10),
        sigma: sigma,
    }
  }

  train(epochs) {
      const ll = [];
      for(let i = 0; i < epochs; i++) {
          // E-Step
          const gamma = tf.div(tf.mul(this.gaus1.prob, tf.exp(tf.neg(this.data.sub(this.gaus1.mu).pow(2)).div(tf.mul(2, this.gaus1.sigma)))), 
          tf.mul(this.gaus1.prob, tf.exp(tf.neg(this.data.sub(this.gaus1.mu).pow(2)).div(tf.mul(2, this.gaus1.sigma)))).add(tf.mul(1-this.gaus1.prob, tf.exp(tf.neg(this.data.sub(this.gaus2.mu).pow(2)).div(tf.mul(2, this.gaus2.sigma))))));

          // M-Step 1
          this.gaus1.mu = tf.sum(tf.sub(1, gamma).transpose().dot(this.data), 0).div(tf.sum(tf.sub(1, gamma), 0));
          // M-Step 2
          this.gaus2.mu = tf.sum(gamma.transpose().dot(this.data), 0).div(tf.sum(gamma, 0));
          // M-Step 3
          this.gaus1.sigma = tf.sum(tf.sub(1, gamma).transpose().dot(this.data.sub(this.gaus1.mu).pow(2))).div(tf.sum(tf.sub(1, gamma)));
          // M-Step 4
          this.gaus2.sigma = tf.sum(gamma.transpose().dot(this.data.sub(this.gaus2.mu).pow(2))).div(tf.sum(gamma));
          // M-Step 5
          this.gaus1.prob = tf.sum(gamma, 0).div(this.data.shape[0]).dataSync()[0];

          // Log likelihood
          ll[i] = tf.sum(tf.log(tf.mul(this.gaus1.prob, tf.exp(tf.neg(this.data.sub(this.gaus1.mu).pow(2)).div(tf.mul(2, this.gaus1.sigma)))).add(tf.mul(1-this.gaus1.prob, tf.exp(tf.neg(this.data.sub(this.gaus2.mu).pow(2)).div(tf.mul(2, this.gaus2.sigma)))))));

          console.log('Log likelihood: ' + ll[i].dataSync(), this.gaus1.prob);
      }
  }
}

