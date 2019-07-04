console.log('Welcome, feel free to dig this.');
let sizeh  = Math.floor(window.innerHeight/5);
let sizew = sizeh;
let sizeImage = sizeh*sizew;

let nH, nW, nImage;
let mask;

// settings of nnet:
let networkSize = 8; //16
let nHidden = 4; //8
let nOut = 3; // r, g, b layers

// support variables:
let img;
let img2;
let G = new R.Graph(false);

let initModel = function() {
  "use strict";

  let model = [];
  let i;

  let randomSize = 1.0;

  // define the model below:
  model.w_in = R.RandMat(networkSize, 3, 0, randomSize); // y, x, and bias(Population mean for initialization, Standard deviation for initialization)

  for (i = 0; i < nHidden; i++) {
    model['w_'+i] = R.RandMat(networkSize, networkSize, 0, randomSize);
  }

  model.w_out = R.RandMat(nOut, networkSize, 0, randomSize); // output layer

  return model;
};


let forwardNetwork = function(G, model, x_, y_) {
  // x_, y_ is a normal javascript float, will be converted to a mat object below
  // G is a graph to amend ops to
  let x = new R.Mat(3, 1); // input
  let i;
  x.set(0, 0, x_);
  x.set(1, 0, y_);
  x.set(2, 0, 1); // bias input.
  let out;
  out = G.tanh(G.mul(model.w_in, x));
  for (i = 0; i < nHidden; i++) {
    out = G.tanh(G.mul(model['w_'+i], out));
  }
  out = G.sigmoid(G.mul(model.w_out, out));
  return out;
};

function getColorAt(model, x, y) {
  // function that returns a color given coordintes (x, y)
  // (x, y) are scaled to -0.5 -> 0.5 for image recognition later
  // but it can be behond the +/- 0.5 for generation above and beyond
  // recognition limits
  let r, g, b;
  let out = forwardNetwork(G, model, x, y);

  r = out.w[0]*255.0;
  g = out.w[1]*255.0;
  b = out.w[2]*255.0;    
    
  return color(r, g, b);
}

function genImage(img, model) {
  let i, j, m, n;
  img.loadPixels();
  for (i = 0, m=img.width; i < m; i++) {
    for (j = 0, n=img.height; j < n; j++) {
      img.set(i, j, getColorAt(model, i/sizeh-0.5,j/sizew-0.5));
    }
  }
  img.updatePixels();
}

function setup() {

  "use strict";
  let myCanvas;
  myCanvas = createCanvas(windowWidth,windowHeight);
  myCanvas.parent('container');
    
  nW = 5;
  nH = 5;    
  nImage = nH*nW;
  mask = R.zeros(nImage);

  img = createImage(sizeh, sizew);

  frameRate(60);
}

function getRandomLocation() {
  let i, result=0, r;
  for (i=0;i<nImage;i++) {
    result += mask[i];
  }
  if (result === nImage) {
    mask = R.zeros(nImage);
  }
  do {
    r = R.randi(0, nImage);
  } while (mask[r] !== 0);
  mask[r] = 1;
  return r;
}

function displayImage(n) {
  let row = Math.floor(n/nW);
  let col = n % nW;
  image(img, col*sizew, row*sizeh);
}

function draw() {
  model = initModel();
  genImage(img, model);
  displayImage(getRandomLocation());
  updateVal();
    updateWrapper();
}

function updateVal(){
    networkSize = select('#networkSize').value();
    select('#networkText').html('Network size: ' + networkSize);
    nHidden = select('#hiddenNum').value();
let networkDepth = nHidden + 1 ;
    select('#hiddenText').html('Network depth: ' + networkDepth +`<br>(Hidden layers: ${nHidden})`);    
}

function updateWrapper(){
    let wrapperPos = sizew*nW + ((window.innerWidth-sizew*nW)/2) - 160 ; 
    let wrapper = document.querySelector('.wrapper');
    wrapper.style.left = `${wrapperPos}px`;
}

