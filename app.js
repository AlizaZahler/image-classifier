const fileInput = document.querySelector('#file');
const upload = document.querySelector('#upload');
const preview = document.querySelector('#preview');
const status = document.querySelector('#status');
const result = document.querySelector('#result');
const progress = document.querySelector('#progress');
const names = {'2d': '2D Image', '3d': '3D Image', live: 'Live'};
let objectURL;
let busy = false;
upload.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) analyze(fileInput.files[0]); fileInput.value = ''; });
for (const event of ['dragenter', 'dragover']) upload.addEventListener(event, e => { e.preventDefault(); if (!busy) upload.classList.add('drag'); });
for (const event of ['dragleave', 'drop']) upload.addEventListener(event, e => { e.preventDefault(); upload.classList.remove('drag'); });
upload.addEventListener('drop', e => { if (!busy && e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]); });
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());
async function analyze(file) {
  if (busy) return;
  status.classList.remove('error');
  for (const label of Object.keys(names)) document.querySelector(`#score-${label}`).textContent = '—';
  document.querySelectorAll('.score').forEach(row => row.classList.remove('winner'));
  result.textContent = '';
  progress.className = 'progress';
  progress.setAttribute('aria-valuenow', '0');
  preview.hidden = true;
  document.querySelector('#icon').removeAttribute('hidden');
  if (objectURL) { URL.revokeObjectURL(objectURL); objectURL = null; }
  if (file.size > 20 * 1024 * 1024) {
    status.textContent = 'Choose an image smaller than 20 MB.';
    status.classList.add('error');
    return;
  }
  busy = true;
  upload.disabled = true;
  upload.setAttribute('aria-busy', 'true');
  objectURL = URL.createObjectURL(file);
  preview.src = objectURL;
  preview.hidden = false;
  preview.onerror = () => { preview.hidden = true; document.querySelector('#icon').removeAttribute('hidden'); };
  document.querySelector('#icon').setAttribute('hidden', '');
  document.querySelector('#upload-label').textContent = 'Choose another image';
  progress.className = 'progress busy';
  progress.removeAttribute('aria-valuenow');
  status.textContent = 'Analyzing your image…';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 90000);
  try {
    const response = await fetch('/predict', {method: 'POST', body: file, signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not analyze the image.');
    for (const label of Object.keys(names)) document.querySelector(`#score-${label}`).textContent = `${(data.scores[label] * 100).toFixed(2)}%`;
    document.querySelector(`[data-class="${data.label}"]`).classList.add('winner');
    result.textContent = `Prediction: ${names[data.label]}`;
    status.textContent = 'Analysis complete.';
    progress.className = 'progress complete';
    progress.setAttribute('aria-valuenow', '100');
  } catch (error) {
    result.textContent = 'Try another image.';
    status.textContent = error.name === 'AbortError' ? 'Analysis timed out. Please try again.' : error.message;
    status.classList.add('error');
    progress.className = 'progress';
    progress.setAttribute('aria-valuenow', '0');
  } finally {
    clearTimeout(timeout);
    busy = false;
    upload.disabled = false;
    upload.removeAttribute('aria-busy');
  }
}
