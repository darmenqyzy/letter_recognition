CLASSES = {
    0: 'Letter A',
    1: 'Letter B',
    2: 'Letter C',
    3: 'Letter D',
    4: 'Letter E',
    5: 'Letter F',
    6: 'Letter G',
    7: 'Letter H',
    8: 'Letter I',
    9: 'Letter J',
    10: 'Letter K',
    11: 'Letter L',
    12: 'Letter M',
    13: 'Letter N',
    14: 'Letter O',
    15: 'Letter P',
    16: 'Letter Q',
    17: 'Letter R',
    18: 'Letter S',
    19: 'Letter T',
    20: 'Letter U',
    21: 'Letter V',
    22: 'Letter W',
    23: 'Letter X',
    24: 'Letter Y',
    25: 'Letter Z'
  };
const MODEL_PATH ='model.json';
const IMAGE_SIZE = 192;
const TOPK_PREDICTIONS = 26;
let my_model;
const demo = async () => {
    status('Loading model...');
    my_model = await tf.loadLayersModel(MODEL_PATH);
    my_model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
    status('');
    const letterElement = document.getElementById('letter');
    if (letterElement.complete && letterElement.naturalHeight !== 0) {
      predict(letterElement);
      letterElement.style.display = '';
    } else {
      letterElement.onload = () => {
        predict(letterElement);
        letterElement.style.display = '';
      }
    }
    document.getElementById('file-container').style.display = '';
  };

async function predict(imgElement) {
    status('Predicting...');
    const startTime1 = performance.now();
    let startTime2;
    const logits = tf.tidy(() => {
      const img = tf.browser.fromPixels(imgElement).toFloat();
      const normalized = img.div(255.0);
      const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
      startTime2 = performance.now();
      return my_model.predict(batched);
    });
  
    const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
    const totalTime1 = performance.now() - startTime1;
    const totalTime2 = performance.now() - startTime2;
    status(`Done in ${Math.floor(totalTime1)} ms ` + `(preprocessing: ${Math.floor(totalTime2)} ms)`);
    showResults(imgElement, classes);
  }
  
  async function getTopKClasses(logits, topK) {
    const values = await logits.data();
  
    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }
  
    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
      topClassesAndProbs.push({
        className: CLASSES[topkIndices[i]],
        probability: topkValues[i]
      })
    }
    return topClassesAndProbs;
  }
  
  function showResults(imgElement, classes) {
    const predictionContainer = document.createElement('div');
    predictionContainer.className = 'pred-container';
    const imgContainer = document.createElement('div');
    imgContainer.appendChild(imgElement);
    predictionContainer.appendChild(imgContainer);
  
    const probsContainer = document.createElement('div');
    for (let i = 0; i < 1; i++) {
      const row = document.createElement('div');
      row.className = 'row';
      const classElement = document.createElement('div');
      classElement.className = 'cell';
      classElement.innerText = classes[i].className;
      row.appendChild(classElement);
      classElement.style.fontSize = '20px';
      classElement.style.paddingTop = '80px';
      classElement.style.paddingLeft = '20px';
      classElement.style.fontWeight = 'bold';
      classElement.style.color = 'black';
            
      const probsElement = document.createElement('div');
      probsElement.className = 'cell';
      row.appendChild(probsElement);
  
      probsContainer.appendChild(row);
    }
    predictionContainer.appendChild(probsContainer);
    predictionsElement.insertBefore(
        predictionContainer, predictionsElement.firstChild);
  }
  
  const filesElement = document.getElementById('files');
  filesElement.addEventListener('change', evt => {
    let files = evt.target.files;
    for (let i = 0, f; f = files[i]; i++) {
      if (!f.type.match('image.*')) {
        continue;
      }
      let reader = new FileReader();
      const idx = i;
      reader.onload = e => {
        let img = document.createElement('img');
        img.src = e.target.result;
        img.width = IMAGE_SIZE;
        img.height = IMAGE_SIZE;
        img.onload = () => predict(img);
      };
      reader.readAsDataURL(f);
    }
  });
  
const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const predictionsElement = document.getElementById('predictions');
  
demo();