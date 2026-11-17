import './viewer.js';

const $ = (selector) => document.querySelector(selector);
const palette = ['#80b7d9', '#ba9ce0', '#81c8a4', '#e0b77c', '#df8e9c', '#8acbd0'];
const paths = {
  play: 'M8 5l10 7-10 7z', pause: 'M8 5v14M16 5v14', back: 'M6 5v14M18 5l-9 7 9 7z',
  forward: 'M18 5v14M6 5l9 7-9 7z', plus: 'M12 5v14M5 12h14',
  undo: 'M8 4L3 9l5 5M3 9h11a6 6 0 010 12', redo: 'M16 4l5 5-5 5M21 9H10a6 6 0 000 12',
  trash: 'M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7M14 10v7',
  more: 'M5 9l7 7 7-7', close: 'M6 6l12 12M18 6L6 18',
};
function icon(element, name) {
  element.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${paths[name]}"></path></svg>`;
}
document.querySelectorAll('[data-icon]').forEach((button) => icon(button, button.dataset.icon));
const emit = (name, detail) => window.dispatchEvent(new CustomEvent(`goddard:${name}`, { detail }));
const clone = (value) => structuredClone(value);
const sessions = new Map();
let session = null;
let data = null;
let frame = 0;
let cursor = 0;
let playing = false;
let lastTime = 0;
let selected = null;
let activePerformer = 'blue';
let gesture = null;
let loading = false;
let saving = false;
let loadVersion = 0;
const fileLists = {
  pending: { version: 0, next: null, files: [], container: '#files', count: '#file-count', more: '#more' },
  done: { version: 0, next: null, files: [], container: '#done-files', count: '#done-count', more: '#done-more' },
};
let timelineWidth = 900;
let labelWidth = 180;
let statusTimer;

function status(message = '', error = false) {
  clearTimeout(statusTimer);
  $('#status').textContent = message;
  $('#status').classList.toggle('error', error);
  if (message && !error) statusTimer = setTimeout(() => status(), 2500);
}
window.addEventListener('goddard:viewer-error', (event) => status(event.detail, true));
async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error ?? `Request failed (${response.status})`);
  }
  return response;
}
function dirty(item) { return item && JSON.stringify(item.document) !== item.saved; }
function refreshButtons() {
  const unavailable = !session || loading;
  for (const id of ['play', 'back', 'forward', 'add-skill']) $(`#${id}`).disabled = unavailable;
  $('#save').disabled = unavailable || saving;
  $('#save').classList.toggle('dirty', Boolean(dirty(session)));
  $('#undo').disabled = unavailable || !session.undo.length;
  $('#redo').disabled = unavailable || !session.redo.length;
  $('#delete').disabled = unavailable || !selected;
  const clip = selectedClip();
  for (const button of document.querySelectorAll('[data-performer]')) {
    button.disabled = unavailable || Boolean(gesture);
    button.setAttribute('aria-pressed', String(button.dataset.performer === activePerformer));
    if (button.dataset.performer === 'unassigned') {
      button.hidden = activePerformer !== 'unassigned'
        && !session?.document.skills.some((skill) => skill.clips.some((item) => !item.performer));
    }
  }
  $('#clip-performer').hidden = !clip;
  $('#clip-performer').disabled = unavailable || Boolean(gesture);
  $('#clip-performer').value = clip?.performer ?? '';
  document.querySelectorAll('.file').forEach((button) => {
    button.classList.toggle('active', button.dataset.file === session?.info.source.file);
    button.classList.toggle('dirty', Boolean(dirty(sessions.get(button.dataset.file))));
  });
}
function selectedClip() {
  return session?.document.skills.find((skill) => skill.id === selected?.skill)
    ?.clips.find((clip) => clip.id === selected?.clip);
}
document.querySelectorAll('[data-performer]').forEach((button) => {
  button.addEventListener('click', () => {
    if (!session || loading || gesture) return;
    if (activePerformer === button.dataset.performer) return;
    activePerformer = button.dataset.performer;
    selected = null;
    renderRows();
  });
});
$('#clip-performer').addEventListener('change', () => {
  if (!session || loading || gesture) return;
  const clip = selectedClip();
  const performer = $('#clip-performer').value;
  if (!clip || !performer || performer === clip.performer) return;
  edit(() => {
    if (performer === 'both') {
      const skill = session.document.skills.find((item) => item.id === selected.skill);
      const other = clip.performer === 'blue' ? 'orange' : 'blue';
      clip.performer = other === 'blue' ? 'orange' : 'blue';
      skill.clips.push({ ...clip, id: crypto.randomUUID(), performer: other });
    } else clip.performer = performer;
    selected = null;
  });
});
function remember(before) {
  if (before === JSON.stringify(session.document)) return;
  session.undo.push(before);
  if (session.undo.length > 100) session.undo.shift();
  session.redo.length = 0;
  refreshButtons();
}
function edit(action) {
  if (!session || loading) return;
  const before = JSON.stringify(session.document);
  action();
  remember(before);
  renderRows();
  renderColors();
}
function history(redo = false) {
  if (!session || gesture || loading) return;
  const from = redo ? session.redo : session.undo;
  const to = redo ? session.undo : session.redo;
  if (!from.length) return;
  to.push(JSON.stringify(session.document));
  session.document = JSON.parse(from.pop());
  selected = null;
  renderRows();
  renderColors();
  refreshButtons();
}

async function listFiles(append = false, state = 'pending') {
  const list = fileLists[state];
  const version = ++list.version;
  const offset = append ? list.next : 0;
  if (offset === null) return;
  try {
    const result = await (await request(`/api/replays?q=${encodeURIComponent($('#search').value)}&offset=${offset}&state=${state}`)).json();
    if (version !== list.version) return;
    list.files = append ? [...list.files, ...result.files] : result.files;
    list.displayNames = append ? { ...list.displayNames, ...result.display_names } : result.display_names ?? {};
    list.next = result.next;
    const fragment = document.createDocumentFragment();
    for (const filename of list.files) {
      const button = document.createElement('button');
      button.className = 'file';
      button.dataset.file = filename;
      button.title = filename;
      const text = document.createElement('span');
      text.textContent = list.displayNames[filename] ?? filename.replace(/\.npy$/, '');
      button.append(text);
      button.addEventListener('click', () => selectFile(filename));
      fragment.append(button);
    }
    $(list.container).replaceChildren(fragment);
    $(list.count).textContent = `${list.files.length.toLocaleString()} / ${result.total.toLocaleString()}`;
    $(list.more).hidden = list.next === null;
    refreshButtons();
  } catch (error) { status(error.message, true); }
}
let searchTimer;
function refreshFiles() { return Promise.all([listFiles(), listFiles(false, 'done')]); }
window.addEventListener('focus', refreshFiles);
$('#search').addEventListener('input', () => {
  for (const list of Object.values(fileLists)) ++list.version;
  clearTimeout(searchTimer);
  searchTimer = setTimeout(refreshFiles, 180);
});
$('#more').addEventListener('click', () => listFiles(true));
$('#done-more').addEventListener('click', () => listFiles(true, 'done'));

async function selectFile(filename) {
  if (gesture) return;
  const currentCategories = [session, ...sessions.values()].filter(Boolean)
    .flatMap((item) => item.document.skills.map(({ name, color }) => ({ name, color })));
  const version = ++loadVersion;
  setPlaying(false);
  loading = true;
  refreshButtons();
  status('Loading…');
  try {
    const info = await (await request(`/api/replay?file=${encodeURIComponent(filename)}`)).json();
    const categories = new Map();
    for (const category of [...currentCategories, ...(info.categories ?? [])]) {
      const key = category.name.trim().toLowerCase();
      if (!categories.has(key)) categories.set(key, category);
    }
    if (!session && !categories.size) {
      ['Air dribble', 'Double tap', 'Flip reset'].forEach((name, index) => categories.set(name, { name, color: palette[index] }));
    }
    const buffer = await (await request(`/api/frames?file=${encodeURIComponent(filename)}&sha256=${info.source.sha256}`)).arrayBuffer();
    if (version !== loadVersion) return;
    if (buffer.byteLength !== info.source.frame_count * info.scene_size * 4) throw new Error('Incomplete replay data');
    let item = sessions.get(filename);
    if (item && item.info.source.sha256 !== info.source.sha256) throw new Error('Replay changed; reload this page');
    if (!item || !dirty(item)) {
      item = { info, document: info.document, saved: JSON.stringify(info.document),
        revision: info.revision, undo: [], redo: [], frame: 0 };
      sessions.set(filename, item);
    }
    const wasDirty = dirty(item);
    const before = JSON.stringify(item.document);
    const names = new Set(item.document.skills.map((skill) => skill.name.trim().toLowerCase()));
    for (const [key, { name, color }] of categories) {
      if (!names.has(key)) item.document.skills.push({ id: crypto.randomUUID(), name, color, clips: [] });
    }
    if (!wasDirty) item.saved = JSON.stringify(item.document);
    else if (before !== JSON.stringify(item.document)) {
      item.undo.push(before);
      if (item.undo.length > 100) item.undo.shift();
      item.redo.length = 0;
    }
    session = item;
    data = new Float32Array(buffer);
    selected = null;
    $('#filename').textContent = item.document.display_name ?? filename;
    $('#filename').title = filename;
    $('#empty-view').hidden = true;
    $('#zoom').value = '1';
    $('#timeline-scroll').scrollLeft = 0;
    $('#camera').replaceChildren(new Option('Overview', 'overview'));
    for (let car = 1; car <= info.team_size * 2; car++) $('#camera').add(new Option(`Car ${car}`, String(car)));
    setCamera('1');
    renderRows();
    renderColors();
    resizeTimeline();
    setFrame(item.frame);
    status();
  } catch (error) {
    if (version === loadVersion) status(error.message, true);
  } finally {
    if (version === loadVersion) { loading = false; refreshButtons(); }
  }
}

function seconds(value) {
  const minutes = Math.floor(value / 60);
  return `${minutes}:${(value % 60).toFixed(2).padStart(5, '0')}`;
}
function setFrame(value, keepCursor = false) {
  if (!session || !data) return;
  frame = Math.max(0, Math.min(session.info.source.frame_count - 1, Math.floor(value)));
  if (!keepCursor) cursor = frame;
  session.frame = frame;
  $('#time').textContent = seconds(frame / session.info.fps);
  $('#frame').textContent = `${frame} / ${session.info.source.frame_count - 1}`;
  $('#frame').dataset.frame = String(frame);
  $('#playhead').style.left = `${labelWidth + frame / session.info.source.frame_count * timelineWidth}px`;
  const offset = frame * session.info.scene_size;
  const vector = (start, scale = [1, 1, 1]) => [0, 1, 2].map((i) => data[offset + start + i] * scale[i]);
  const position = (start) => vector(start, [4108, 6000, 2076]);
  const cars = [];
  for (let index = 0; index < session.info.team_size * 2; index++) {
    const start = 9 + 21 * index;
    cars.push({ id: index + 1, team: Number(index >= session.info.team_size),
      physics: { position: position(start), forward: vector(start + 9), up: vector(start + 12) },
      boost: data[offset + start + 15] * 100, demoed: data[offset + start + 17] > .5,
      boosting: data[offset + start + 20] > .5,
      jumping: (session.info.jump_holds?.[index] ?? []).some(([start, end]) => start <= frame && frame < end) });
  }
  emit('frame', { episode: 0, ball: { position: position(0) }, cars, pads: [], controls: [] });
}
function setPlaying(value) {
  playing = Boolean(value && session && !loading);
  if (playing && frame === session.info.source.frame_count - 1) setFrame(0);
  lastTime = performance.now();
  icon($('#play'), playing ? 'pause' : 'play');
  $('#play').setAttribute('aria-label', playing ? 'Pause' : 'Play');
}
function animate(now) {
  if (playing && session) {
    cursor += Math.min((now - lastTime) / 1000, .25) * session.info.fps * Number($('#speed').value);
    if (Math.floor(cursor) !== frame) setFrame(cursor, true);
    if (cursor >= session.info.source.frame_count - 1) setPlaying(false);
  }
  lastTime = now;
  if (gesture) {
    const scroll = $('#timeline-scroll');
    const rect = scroll.getBoundingClientRect();
    const left = rect.left + labelWidth;
    const distance = gesture.x < left + 25 ? gesture.x - left - 25
      : gesture.x > rect.right - 25 ? gesture.x - rect.right + 25 : 0;
    if (distance) {
      scroll.scrollLeft += Math.max(-18, Math.min(18, distance / 3));
      updateGesture();
    }
  }
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
$('#play').addEventListener('click', () => setPlaying(!playing));
$('#back').addEventListener('click', () => { setPlaying(false); setFrame(frame - 1); });
$('#forward').addEventListener('click', () => { setPlaying(false); setFrame(frame + 1); });
function setCamera(value) {
  $('#camera').value = value;
  emit('camera', value === 'overview' ? 'overview' : Number(value));
}
$('#camera').addEventListener('change', () => setCamera($('#camera').value));
window.addEventListener('keydown', (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing) return;
  const camera = { '0': 'overview', '1': '1', '2': '2' }[event.key];
  if (!camera || !session) return;
  event.preventDefault();
  setCamera(camera);
}, { capture: true });

function renderColors() {
  if (!session) return;
  const viewer = session.document.viewer ??= { car_colors: {}, ball_color: '#f0eee4' };
  viewer.car_colors ??= {};
  const fragment = document.createDocumentFragment();
  for (let car = 0; car <= session.info.team_size * 2; car++) {
    const label = document.createElement('label');
    label.className = 'car-color';
    label.textContent = car ? String(car) : '●';
    const input = document.createElement('input');
    input.type = 'color';
    input.title = car ? `Car ${car} color` : 'Ball color';
    input.setAttribute('aria-label', input.title);
    input.value = car ? viewer.car_colors[car] ?? (car <= session.info.team_size ? '#338fff' : '#ff922b') : viewer.ball_color ?? '#f0eee4';
    input.addEventListener('change', () => edit(() => {
      if (car) viewer.car_colors[car] = input.value;
      else viewer.ball_color = input.value;
    }));
    label.append(input);
    fragment.append(label);
  }
  $('#car-colors').replaceChildren(fragment);
  emit('colors', viewer);
}

function renderRows() {
  $('#skill-rows').replaceChildren();
  if (!session) return;
  for (const skill of session.document.skills) {
    const row = document.createElement('div');
    row.className = 'track-row';
    const label = document.createElement('div');
    label.className = 'track-label';
    const color = document.createElement('input');
    color.type = 'color';
    color.value = skill.color;
    color.title = `${skill.name} color`;
    color.setAttribute('aria-label', color.title);
    color.addEventListener('change', () => edit(() => { skill.color = color.value; }));
    const name = document.createElement('input');
    name.className = 'skill-name';
    name.value = skill.name;
    name.setAttribute('aria-label', 'Skill name');
    name.addEventListener('change', () => {
      if (!name.value.trim()) { name.value = skill.name; return; }
      edit(() => { skill.name = name.value.trim(); });
    });
    name.addEventListener('keydown', (event) => { if (event.key === 'Enter') name.blur(); });
    const remove = document.createElement('button');
    remove.className = 'remove-skill';
    remove.title = `Delete ${skill.name} track`;
    remove.setAttribute('aria-label', remove.title);
    icon(remove, 'close');
    remove.addEventListener('click', () => edit(() => {
      session.document.skills = session.document.skills.filter((item) => item.id !== skill.id);
      if (selected?.skill === skill.id) selected = null;
    }));
    label.append(color, name, remove);
    const lane = document.createElement('div');
    lane.className = 'lane skill-lane';
    lane.dataset.skill = skill.id;
    lane.setAttribute('aria-label', `${skill.name} timeline`);
    lane.addEventListener('pointerdown', (event) => beginGesture(event, skill, lane));
    row.append(label, lane);
    $('#skill-rows').append(row);
    drawClips(skill, lane);
  }
  refreshButtons();
}
function drawClips(skill, lane) {
  lane.replaceChildren();
  lane.classList.toggle('both-view', activePerformer === 'both');
  for (const clip of skill.clips) {
    const visible = activePerformer === 'both' ? ['blue', 'orange'].includes(clip.performer)
      : (clip.performer ?? 'unassigned') === activePerformer;
    if (!visible) continue;
    const element = document.createElement('div');
    element.className = `clip${selected?.clip === clip.id ? ' selected' : ''}`;
    element.dataset.clip = clip.id;
    element.style.left = `${clip.start_frame / session.info.source.frame_count * 100}%`;
    element.style.width = `${(clip.end_frame - clip.start_frame) / session.info.source.frame_count * 100}%`;
    element.style.setProperty('--clip-color', skill.color);
    element.dataset.performer = clip.performer ?? 'unassigned';
    element.title = `${skill.name} · ${clip.start_frame}–${clip.end_frame - 1} · ${clip.performer ?? 'Unassigned'}`;
    if (typeof clip.note === 'string' && clip.note) element.title += `\n${clip.note}`;
    const marker = document.createElement('i');
    marker.className = `performer-dot ${clip.performer ?? 'unassigned'}`;
    element.append(marker);
    for (const side of ['start', 'end']) {
      const handle = document.createElement('span');
      handle.className = `handle ${side}`;
      handle.dataset.edge = side;
      element.append(handle);
    }
    lane.append(element);
  }
}
$('#add-skill').addEventListener('click', () => {
  edit(() => session.document.skills.push({ id: crypto.randomUUID(), name: 'Skill',
    color: palette[session.document.skills.length % palette.length], clips: [] }));
  const input = $('#skill-rows').lastElementChild?.querySelector('.skill-name');
  input?.scrollIntoView({ block: 'nearest' });
  input?.focus();
  input?.select();
});
function deleteClip() {
  if (!selected) return;
  edit(() => {
    const skill = session.document.skills.find((item) => item.id === selected.skill);
    if (skill) skill.clips = skill.clips.filter((clip) => clip.id !== selected.clip);
    selected = null;
  });
}
$('#delete').addEventListener('click', deleteClip);
$('#undo').addEventListener('click', () => history());
$('#redo').addEventListener('click', () => history(true));

function frameAt(lane, x) {
  const rect = lane.getBoundingClientRect();
  return Math.max(0, Math.min(session.info.source.frame_count - 1,
    Math.floor((x - rect.left) / rect.width * session.info.source.frame_count)));
}
function beginGesture(event, skill, lane) {
  if (!session || loading || event.button !== 0 || gesture) return;
  event.preventDefault();
  setPlaying(false);
  const clip = skill?.clips.find((item) => item.id === event.target.closest('.clip')?.dataset.clip);
  const anchor = frameAt(lane, event.clientX);
  gesture = { skill, lane, displayLane: lane, anchor, x: event.clientX, initialX: event.clientX,
    y: event.clientY, initialY: event.clientY,
    before: JSON.stringify(session.document), clip, original: clip ? clone(clip) : null,
    mode: !skill || (!clip && activePerformer === 'unassigned') ? 'scrub'
      : !clip ? 'create' : event.target.dataset.edge ?? 'move',
    moved: false, pointer: event.pointerId };
  selected = clip ? { skill: skill.id, clip: clip.id } : null;
  document.body.classList.add('dragging');
  lane.setPointerCapture(event.pointerId);
  setFrame(anchor);
  if (skill) drawClips(skill, lane);
  refreshButtons();
}
function updateGesture() {
  if (!gesture) return;
  const g = gesture;
  const point = frameAt(g.lane, g.x);
  setFrame(point);
  g.moved ||= Math.abs(g.x - g.initialX) >= 3
    || (g.mode === 'move' && Math.abs(g.y - g.initialY) >= 3);
  if (!g.moved || g.mode === 'scrub') return;
  if (g.mode === 'create') {
    if (!g.clip) {
      g.clip = { id: crypto.randomUUID(), start_frame: g.anchor, end_frame: g.anchor + 1,
        performer: activePerformer === 'both' ? 'blue' : activePerformer };
      g.skill.clips.push(g.clip);
      if (activePerformer === 'both') {
        g.other = { ...g.clip, id: crypto.randomUUID(), performer: 'orange' };
        g.skill.clips.push(g.other);
      }
      selected = { skill: g.skill.id, clip: g.clip.id };
    }
    g.clip.start_frame = Math.min(g.anchor, point);
    g.clip.end_frame = Math.max(g.anchor, point) + 1;
    if (g.other) {
      g.other.start_frame = g.clip.start_frame;
      g.other.end_frame = g.clip.end_frame;
    }
  } else if (g.mode === 'move') {
    // Pointer capture stays on the original lane; hit-test the actual pointer
    // position to preview moving this one clip into another skill.
    const destination = document.elementFromPoint(g.x, g.y)?.closest('.skill-lane');
    g.displayLane.classList.remove('drop-target');
    if (destination) {
      const skill = session.document.skills.find((item) => item.id === destination.dataset.skill);
      if (skill && skill !== g.skill) {
        g.skill.clips = g.skill.clips.filter((clip) => clip.id !== g.clip.id);
        drawClips(g.skill, g.displayLane);
        skill.clips.push(g.clip);
        g.skill = skill;
        g.displayLane = destination;
        selected = { skill: skill.id, clip: g.clip.id };
      }
      g.displayLane.classList.add('drop-target');
    }
    const length = g.original.end_frame - g.original.start_frame;
    const start = Math.max(0, Math.min(session.info.source.frame_count - length,
      g.original.start_frame + (Math.abs(g.x - g.initialX) < 3 ? 0 : point - g.anchor)));
    g.clip.start_frame = start;
    g.clip.end_frame = start + length;
  } else if (g.mode === 'start') {
    g.clip.start_frame = Math.min(point, g.original.end_frame - 1);
  } else if (g.mode === 'end') {
    g.clip.end_frame = Math.max(point + 1, g.original.start_frame + 1);
  }
  drawClips(g.skill, g.displayLane);
}
function endGesture(cancel = false) {
  if (!gesture) return;
  const g = gesture;
  if (cancel) { session.document = JSON.parse(g.before); selected = null; }
  else {
    if (g.skill) g.skill.clips.sort((a, b) => a.start_frame - b.start_frame);
    remember(g.before);
  }
  gesture = null;
  if (g.lane.hasPointerCapture(g.pointer)) g.lane.releasePointerCapture(g.pointer);
  document.body.classList.remove('dragging');
  renderRows();
}
window.addEventListener('pointermove', (event) => {
  if (gesture && event.pointerId === gesture.pointer) {
    gesture.x = event.clientX; gesture.y = event.clientY; updateGesture();
  }
});
window.addEventListener('pointerup', (event) => {
  if (gesture && event.pointerId === gesture.pointer) {
    gesture.x = event.clientX; gesture.y = event.clientY; updateGesture(); endGesture();
  }
});
window.addEventListener('pointercancel', () => endGesture(true));
$('#ruler').addEventListener('pointerdown', (event) => beginGesture(event, null, $('#ruler')));

function resizeTimeline() {
  labelWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--label-width'));
  timelineWidth = Math.max(120, $('#timeline-scroll').clientWidth - labelWidth) * Number($('#zoom').value);
  $('#tracks').style.setProperty('--timeline-width', `${timelineWidth}px`);
  $('#ruler').replaceChildren();
  if (!session) { $('#playhead').hidden = true; return; }
  $('#playhead').hidden = false;
  const duration = session.info.source.frame_count / session.info.fps;
  const desired = duration / (timelineWidth / 100);
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(desired, .001)));
  const step = [1, 2, 5, 10].map((n) => n * magnitude).find((n) => n >= desired);
  $('#tracks').style.setProperty('--tick-width', `${step / duration * timelineWidth}px`);
  for (let time = 0; time < duration; time += step) {
    const tick = document.createElement('div');
    tick.className = 'tick';
    tick.style.left = `${time / duration * 100}%`;
    const text = document.createElement('span');
    text.textContent = seconds(time);
    tick.append(text);
    $('#ruler').append(tick);
  }
  setFrame(frame);
}
new ResizeObserver(resizeTimeline).observe($('#timeline-scroll'));
$('#zoom').addEventListener('input', () => {
  resizeTimeline();
  if (session) $('#timeline-scroll').scrollLeft = Math.max(0,
    frame / session.info.source.frame_count * timelineWidth - ($('#timeline-scroll').clientWidth - labelWidth) / 2);
});
$('#divider').addEventListener('pointerdown', (event) => {
  const divider = event.currentTarget;
  divider.setPointerCapture(event.pointerId);
  const move = (e) => document.documentElement.style.setProperty('--editor-height', `${Math.max(130, Math.min(innerHeight - 180, innerHeight - e.clientY))}px`);
  const stop = () => { divider.removeEventListener('pointermove', move); divider.removeEventListener('pointerup', stop); };
  divider.addEventListener('pointermove', move);
  divider.addEventListener('pointerup', stop);
});
$('#divider').addEventListener('keydown', (event) => {
  if (!['ArrowUp', 'ArrowDown'].includes(event.key)) return;
  event.preventDefault();
  const height = $('#editor').clientHeight + (event.key === 'ArrowUp' ? 30 : -30);
  document.documentElement.style.setProperty('--editor-height', `${Math.max(130, Math.min(innerHeight - 180, height))}px`);
});

async function save() {
  if (!session || loading || saving || gesture) return;
  const item = session;
  const saved = JSON.stringify(item.document);
  saving = true;
  refreshButtons();
  try {
    const result = await (await request('/api/labels', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file: item.info.source.file, document: JSON.parse(saved), revision: item.revision }),
    })).json();
    item.revision = result.revision;
    item.saved = saved;
    for (const button of $('#files').querySelectorAll('.file')) {
      if (button.dataset.file === item.info.source.file) button.remove();
    }
    if (session === item) { status('Saved'); $('#save').title = `${result.label_file} · Ctrl+S`; }
    await refreshFiles();
  } catch (error) { status(error.message, true); }
  finally { saving = false; refreshButtons(); }
}
$('#save').addEventListener('click', save);
window.addEventListener('keydown', (event) => {
  const text = event.target.closest('input, select, textarea');
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
    event.preventDefault();
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
    save();
    return;
  }
  if (text || !session) return;
  if (event.code === 'Space' && event.target.closest('button')) return;
  if ((event.ctrlKey || event.metaKey) && ['z', 'y'].includes(event.key.toLowerCase())) {
    event.preventDefault(); history(event.shiftKey || event.key.toLowerCase() === 'y');
  } else if (event.code === 'Space') {
    event.preventDefault(); if (!event.repeat) setPlaying(!playing);
  } else if (['ArrowLeft', 'ArrowRight'].includes(event.key)) {
    event.preventDefault(); setPlaying(false); setFrame(frame + (event.key === 'ArrowLeft' ? -1 : 1) * (event.shiftKey ? 10 : 1));
  } else if (['Delete', 'Backspace'].includes(event.key)) {
    event.preventDefault(); if (!gesture) deleteClip();
  } else if (event.key === 'Escape') endGesture(true);
});
window.addEventListener('beforeunload', (event) => {
  if ([...sessions.values()].some(dirty)) { event.preventDefault(); event.returnValue = ''; }
});
resizeTimeline();
refreshFiles();
