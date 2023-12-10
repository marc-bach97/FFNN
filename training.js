
var number_of_hidden_layers=1;
var model;
const input_graph = document.getElementById('input_graph');
const training_graph = document.getElementById('training_graph');
const testing_graph = document.getElementById('testing_graph');
var mode_trained=false;
var input_data;

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

$('#add_hidden_layer').click(function() {
    $("#hidden_layers").append(
        `<div class="row hidden_layer_item" id="layer_${number_of_hidden_layers+1}" style="align-items: end;margin-top:10px;">
        <div class="col-sm-2">
          Layer ${number_of_hidden_layers+1}
        </div>
        <div class="col-sm-4">
          <label for="neurons" class="form-label">Neurons</label>
          <input type="number" min="1" step="1" id="neurons" class="form-control" value="5" aria-describedby="Neurons">
        </div>
        <div class="col-sm-4">
          <label for="activation" class="form-label">Activation</label>
          <select id="activation" class="form-control form-select-md" aria-label=".form-select example">
            <option selected value="relu">relu</option>
            <option value="linear">linear</option>
            <option value="elu">elu</option>
            <option value="sigmoid">sigmoid</option>
            <option value="tanh">tanh</option>
          </select>
        </div>
        <div class="col-sm-2">
          <button type="button" class="btn btn-danger color-white" onclick="deleteLayer(${number_of_hidden_layers+1})">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-trash3" viewBox="0 0 16 16">
              <path d="M6.5 1h3a.5.5 0 0 1 .5.5v1H6v-1a.5.5 0 0 1 .5-.5M11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3A1.5 1.5 0 0 0 5 1.5v1H2.506a.58.58 0 0 0-.01 0H1.5a.5.5 0 0 0 0 1h.538l.853 10.66A2 2 0 0 0 4.885 16h6.23a2 2 0 0 0 1.994-1.84l.853-10.66h.538a.5.5 0 0 0 0-1h-.995a.59.59 0 0 0-.01 0zm1.958 1-.846 10.58a1 1 0 0 1-.997.92h-6.23a1 1 0 0 1-.997-.92L3.042 3.5zm-7.487 1a.5.5 0 0 1 .528.47l.5 8.5a.5.5 0 0 1-.998.06L5 5.03a.5.5 0 0 1 .47-.53Zm5.058 0a.5.5 0 0 1 .47.53l-.5 8.5a.5.5 0 1 1-.998-.06l.5-8.5a.5.5 0 0 1 .528-.47ZM8 4.5a.5.5 0 0 1 .5.5v8.5a.5.5 0 0 1-1 0V5a.5.5 0 0 1 .5-.5"/>
            </svg>
          </button>
        </div>
      </div>`
     );
     number_of_hidden_layers++;
});

function deleteLayer(layer_id){
    var layer_ele_id="#layer_"+layer_id;
    $(layer_ele_id).remove();
}

function generateData(){
    var n_samples=$('#number_of_samples').find(":selected").val();
    if(n_samples==0){
        endOperation();
        alert('Please Select the number of samples');
        return;
    }
    var xValues = generateRandomXValues(n_samples);
    var yValues=xValues.map(calculateYValue);
    var series_title;
    input_data=[];

    if($('#add_noise').is(":checked")){
        for (let i = 0; i < n_samples; i++) {
            input_data.push({ x: xValues[i], y: yValues[i]+gaussianRandom() });
          }
          series_title="Data Noisy";
    }else{
        for (let i = 0; i < n_samples; i++) {
            input_data.push({ x: xValues[i], y: yValues[i] });
          }
          series_title="Data without Noise";

    }

    

    tfvis.render.scatterplot(
        input_graph,
        {values: [input_data], series: [series_title]},
        {xLabel: 'X'},
        { yLabel: 'Y'},      
    );

}

function generateTestingData(){
    var n_samples=Math.floor(Math.random() * (200 - 50 + 1) ) +50;
    
    var xValues = generateRandomXValues(n_samples);
    var yValues=xValues.map(calculateYValue);
    var testing_data_x=[];
    var testing_data_y=[];

    if($('#add_noise').is(":checked")){
        for (let i = 0; i < n_samples; i++) {
            testing_data_x.push(xValues[i]);
            testing_data_y.push(yValues[i]+gaussianRandom());

          }
    }else{
        for (let i = 0; i < n_samples; i++) {
            testing_data_x.push(xValues[i]);
            testing_data_y.push(yValues[i]);
          }

    }

    return [testing_data_x,testing_data_y];

}
function startOperation(){
    $('#gear_icon').addClass('icn-spinner');
    $('#generate_model').addClass('disabled');
    $('#save_model').addClass('disabled');

}
function endOperation(){
    $('#generate_model').removeClass('disabled');
    $('#save_model').removeClass('disabled');
    $('#gear_icon').removeClass('icn-spinner');
}

function getHiddenLayers(){
    var hidden_layers=[];
    $('.hidden_layer_item').each(function(i, obj) {
        var neurons=$(obj).find('#neurons').val();
        var activation=$(obj).find('#activation').find(":selected").val();
        hidden_layers.push({ neurons:Number(neurons), activation: activation });

    });
    hidden_layers.forEach(function(layer) {
        console.log(layer['neurons'],layer['activation']);
    });
    return hidden_layers;
}


function createModel() {
    const model = tf.sequential();
  
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
    var hidden_layers=getHiddenLayers();
    hidden_layers.forEach(function(layer) {
        model.add(tf.layers.dense({units: layer['neurons'], activation: layer['activation'],useBias: true}));
    });
    model.add(tf.layers.dense({units: 1, useBias: true}));
  
    return model;
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

  async function trainModel(model, inputs, labels,batchSize,epochs) {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });  
    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        training_graph,
        [ 'mse'],
        { height: 250, callbacks: ['onEpochEnd'] }
      )
    });
  }

  function testModel(model,inputData, normalizationData) {

    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
  
    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
  
      const xsNorm = tf.linspace(0, 1, 200);
      const predictions = model.predict(xsNorm.reshape([200, 1]));
  
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


  async function generateModel(){
    var n_samples=$('#number_of_samples').find(":selected").val();
    if(n_samples==0){
        endOperation();
        alert('Bitte wählen Sie Anzahl der Datenproben aus');
        return;
    }
    startOperation();
    generateData();
    model= createModel();
    const tensorData = convertToTensor(input_data);
    const {inputs, labels} = tensorData;
    var  batchSize=Number($('#batch_size').val());
    var  epochs=Number($('#epochs').val());

   await trainModel(model, inputs, labels,batchSize,epochs);
   console.log('Done Training');
   testModel(model,input_data, tensorData);
   endOperation();
   mode_trained=true;
   
   


}
async function saveModel(){
  var file_name=$('#file_name').val();
  if(file_name==""){
    alert('File name is required!');
        return;
  }
  if (!mode_trained){
    alert('No Model yet!');
    return;
  }
  const saveResult = await model.save('downloads://model_'+file_name);
  console.log(saveResult);
  $('#file_name').val="";
  $('#savemodel_dialoge').modal('hide');

}
