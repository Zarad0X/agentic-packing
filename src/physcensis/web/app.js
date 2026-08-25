const promptBox = document.querySelector('#prompt');
const generateButton = document.querySelector('#generate');
const loading = document.querySelector('#loading');
const render = document.querySelector('#render');
const summary = document.querySelector('#summary');
const state = document.querySelector('#scene-state');
let isPhysicalBackend = false;

document.querySelectorAll('[data-prompt]').forEach(button => {
  button.addEventListener('click', () => { promptBox.value = button.dataset.prompt; });
});

fetch('/api/health').then(response => response.json()).then(data => {
  document.querySelector('#backend').textContent = `${data.backend} · ${data.agent}`;
  isPhysicalBackend = data.backend === 'genesis';
  document.querySelector('#viewport-label').textContent = isPhysicalBackend ? 'GENESIS VIEWPORT' : 'GEOMETRY PREVIEW';
  state.textContent = isPhysicalBackend ? 'PHYSICALLY STABLE' : 'GEOMETRY VALID';
  document.querySelector('.verified').textContent = isPhysicalBackend ? '✓ physics verified' : '✓ geometry verified';
});

function number(value, digits = 3) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
}

async function generate() {
  const prompt = promptBox.value.trim();
  if (!prompt) return;
  generateButton.disabled = true;
  loading.classList.add('active');
  state.textContent = 'SOLVING';
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt}),
    });
    const data = await response.json();
    if (!response.ok || !data.success) throw new Error(data.error || data.summary || 'Generation failed');
    if (data.image) render.src = `${data.image}?t=${Date.now()}`;
    document.querySelector('#objects').textContent = data.object_count;
    document.querySelector('#penalty').textContent = number(data.metrics.spatial_penalty);
    document.querySelector('#settle').textContent = `${number((data.metrics.settle_distance_m || 0) * 1000)} mm`;
    document.querySelector('#packing').textContent = data.packing_fraction
      ? `${number(data.packing_fraction * 100, 1)}%`
      : '—';
    document.querySelector('#floor-use').textContent = data.metrics.floor_coverage
      ? `${number(data.metrics.floor_coverage * 100, 1)}%`
      : '—';
    document.querySelector('#organization').textContent = data.metrics.organization_score
      ? `${number(data.metrics.organization_score * 100, 1)}%`
      : '—';
    summary.textContent = data.summary;
    state.textContent = isPhysicalBackend ? 'PHYSICALLY STABLE' : 'GEOMETRY VALID';
    const records = (data.program || []).filter(record => Array.isArray(record) && record.length === 4);
    document.querySelector('#program-code').textContent = records.slice(0, 9).map(record => JSON.stringify(record)).join('\n');
  } catch (error) {
    summary.textContent = error.message;
    state.textContent = 'NEEDS REVISION';
  } finally {
    loading.classList.remove('active');
    generateButton.disabled = false;
  }
}

generateButton.addEventListener('click', generate);
promptBox.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') generate();
});
