function gaussianRandom(mean=0, stdev=0.1) {
    const u = 1 - Math.random(); 
    const v = Math.random();
    const z = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    return z * stdev + mean;
}
function calculateYValue(x){
  return (x+0.8)*(x-0.2)*(x-0.3)*(x-0.6);
}

function generateRandomXValues(N) {
  const xValues = tf.linspace(-1, 1, N).dataSync();
  return Array.from(xValues);
}

function generateData(n_samples,noisy){
    var xValues = generateRandomXValues(n_samples);
    var yValues=xValues.map(calculateYValue);
    var input_data=[];

    if(noisy){
        for (let i = 0; i < n_samples; i++) {
            input_data.push({ x: xValues[i], y: yValues[i]+gaussianRandom() });
          }
    }else{
        for (let i = 0; i < n_samples; i++) {
            input_data.push({ x: xValues[i], y: yValues[i] });
          }

    }
    return input_data;
}

function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.
  
    return tf.tidy(() => {
      // Step 1. Shuffle the data
      tf.util.shuffle(data);
  
      // Step 2. Convert data to Tensor
      const inputs = data.map(d => d.x)
      const labels = data.map(d => d.y);
  
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();
  
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
  
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });
  }

  function testModel(model,inputData, normalizationData,testing_graph) {

    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
  
    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
  
      const xsNorm = tf.linspace(0, 1, 50);
      const predictions = model.predict(xsNorm.reshape([50, 1]));
  
      const unNormXs = xsNorm
        .mul(inputMax.sub(inputMin))
        .add(inputMin);
  
      const unNormPreds = predictions
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
  
      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });
  
  
    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: preds[i]}
    });
  
    const originalPoints = inputData.map(d => ({
      x: d.x, y: d.y,
    }));
  
    tfvis.render.scatterplot(
      testing_graph,
      {values: [originalPoints,predictedPoints], series: ['original', 'predicted']},
      {xLabel: 'X'},
      { yLabel: 'Y'}, 
    );
  }

async function loadModel(results_part,noise){
  
    $.getJSON( "./files/results.json", async function(data) {
      var model_data=data[results_part];
      var n_samples=model_data['n_samples'];
      var batchSize=model_data['batchSize'];
      var epochs=model_data['epochs'];
      var model_name=model_data['file'];
      var hidden_layers=model_data['hidden_layers'];
      document.querySelector("."+results_part).querySelector("#n_samples").value=n_samples;
      document.querySelector("."+results_part).querySelector("#batchSize").value=batchSize;
      document.querySelector("."+results_part).querySelector("#epochs").value=epochs;
      $('.'+results_part+' #hidden_layers_area').empty();
      hidden_layers.forEach(function(layer,index) {
        $('.'+results_part+' #hidden_layers_area').append(
          `<div class="row hidden_layer_item" id="layer_1" style="align-items: end;">
          <div class="col-sm-2">
            Layer ${index+1}
          </div>
          <div class="col-sm-4">
            <label for="neurons" class="form-label">Neurons</label>
            <input type="number"  id="neurons" class="form-control neurons" value="${layer['neurons']}" disabled aria-describedby="Neurons">
          </div>
          <div class="col-sm-4">
            <label for="activation" class="form-label">Activation</label>
            <input type="text"  id="activation" class="form-control neurons" value="${layer['activation']}" disabled aria-describedby="Neurons">
          </div>
        </div>`
       );
      });

    var input_data=generateData(n_samples,noise);
    const tensorData = convertToTensor(input_data);
    const model = await tf.loadLayersModel('./files/'+model_name);
    var testing_graph=document.querySelector("."+results_part).querySelector("#testing_graph");
    testModel(model,input_data, tensorData,testing_graph);


   }).done(function() {
      console.log( "second success" );
   })
   .fail(function() {
     alert('Could not load the files')
   })
   .always(function() {
      console.log( "complete" );
   });
  
  }
