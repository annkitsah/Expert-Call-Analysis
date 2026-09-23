'use strict';

/* All DOM is built with createElement + text nodes (never innerHTML), so transcript
   text uploaded by a user can never inject markup. */

const $ = (selector, root = document) => root.querySelector(selector);

const state = {
  health: null,
  corpus: null,
  segments: new Map(),
  transcripts: new Map(),
  analysis: null,
  tab: 'guide',
  transcriptId: null,
  focus: null, // {segment_id, start, end} highlighted in the Transcripts tab
  drawerEvidence: null,
  busy: false,
  error: null,
  chat: [], // {q, res?, error?, pending}
};

// ---------------------------------------------------------------- helpers ---
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch { /* empty or non-JSON body */ }
  if (!response.ok) {
    const detail = body && body.detail;
    if (typeof detail === 'string') throw new Error(detail);
    if (Array.isArray(detail)) throw new Error(detail.map((d) => d.msg).join('; '));
    throw new Error(`Request failed (${response.status}).`);
  }
  return body;
}

const postJson = (path, data) =>
  api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });

const roleMarket = (t) => [t.role, t.market].filter(Boolean).join(', ');

function setCorpus(corpus) {
  state.corpus = corpus;
  state.segments = new Map();
  state.transcripts = new Map();
  for (const t of corpus.transcripts) {
    state.transcripts.set(t.id, t);
    for (const s of t.segments) state.segments.set(s.id, s);
  }
  if (!state.transcripts.has(state.transcriptId)) state.transcriptId = corpus.transcripts[0]?.id ?? null;
}

function markedText(text, start, end) {
  if (!Number.isInteger(start) || !Number.isInteger(end)) return text;
  return [text.slice(0, start), h('span', { class: 'mark' }, text.slice(start, end)), text.slice(end)];
}

// --------------------------------------------------------- shared widgets ---
function tsButton(ev) {
  return h('button', {
    class: 'ts', type: 'button',
    'aria-label': `Open the source turn at ${ev.timestamp} in ${ev.expert_name}'s transcript`,
    onclick: (e) => openSource(ev, e.currentTarget),
  }, ev.timestamp);
}

function quoteFigure(ev, { who = false } = {}) {
  return h('figure', { class: 'quote' },
    h('blockquote', {}, '\u201C', h('span', { class: 'mark' }, ev.quote), '\u201D'),
    h('figcaption', {}, who ? h('span', { class: 'who' }, `${ev.expert_name}, ${ev.market}`) : null, tsButton(ev)));
}

function emptyState() {
  const n = state.corpus.transcripts.length;
  const canRun = state.health && state.health.llm_configured;
  return h('div', { class: 'empty' },
    h('h2', {}, 'No analysis yet'),
    h('p', {}, `Analyzing reads ${n === 1 ? 'the transcript' : `all ${n} transcripts`} against the guide questions and compares the experts. `
      + 'Every quote is checked against the transcript text before it is shown.'),
    h('button', { class: 'btn primary', type: 'button', disabled: !canRun || state.busy, onclick: () => runAnalysis(false) },
      state.busy ? 'Analyzing\u2026' : `Analyze ${n} ${n === 1 ? 'call' : 'calls'}`));
}

// ------------------------------------------------------------------ views ---
function renderHeader() {
  const c = state.corpus;
  const project = c ? c.guide.title.replace(/^interview guide\s*[\u2013\u2014-]\s*/i, '') : '';
  $('#project').textContent = c ? `${project} \u00B7 ${c.transcripts.length} ${c.transcripts.length === 1 ? 'call' : 'calls'}` : '';
  const btn = $('#analyze-btn');
  const llmReady = state.health && state.health.llm_configured;
  btn.disabled = state.busy || !c || !llmReady;
  const n = c ? c.transcripts.length : 0;
  btn.textContent = state.busy ? 'Analyzing\u2026' : state.analysis ? 'Re-run analysis' : `Analyze ${n} ${n === 1 ? 'call' : 'calls'}`;
  btn.title = llmReady ? '' : 'Set GROQ_API_KEY and restart the server to run analysis.';
}

function renderNotice() {
  const el = $('#notice');
  const noKey = state.health && !state.health.llm_configured;
  if (state.error) {
    el.className = 'notice';
    el.textContent = state.error;
    el.hidden = false;
  } else if (noKey) {
    el.className = 'notice info';
    el.textContent = 'No Groq API key is configured. You can read the transcripts and any cached analysis; set GROQ_API_KEY (see .env.example) and restart to run new analysis or ask questions.';
    el.hidden = false;
  } else {
    el.hidden = true;
  }
}

function renderStatus() {
  const el = $('#statusbar');
  const a = state.analysis;
  if (!a || (state.tab !== 'guide' && state.tab !== 'themes')) { el.hidden = true; return; }
  const dropped = a.quotes_checked - a.quotes_verified;
  const parts = [
    `${a.quotes_verified} of ${a.quotes_checked} quotes matched the transcripts word for word.`,
    dropped > 0 ? `${dropped} that did not match ${dropped === 1 ? 'was' : 'were'} removed.` : null,
    'Timestamps mark where each speaker\u2019s turn starts.',
    `Model: ${a.model}. Generated ${new Date(a.generated_at).toLocaleString()}.`,
  ].filter(Boolean);
  el.replaceChildren(
    parts.join(' '),
    a.warnings.length
      ? h('details', {}, h('summary', {}, `${a.warnings.length} verification ${a.warnings.length === 1 ? 'note' : 'notes'}`),
        h('ul', {}, a.warnings.map((w) => h('li', {}, w))))
      : null);
  el.hidden = false;
}

function answerFor(transcriptId, questionId) {
  const expert = state.analysis.guide_answers.find((g) => g.transcript_id === transcriptId);
  return expert ? expert.answers.find((x) => x.question_id === questionId) : null;
}

function guideCell(t, a) {
  const label = h('div', { class: 'cell-label' }, `${t.expert_name}, ${t.market}`);
  if (!a) return h('div', { class: 'cell' }, label, h('div', { class: 'note warn' }, 'No result for this question.'));
  switch (a.status) {
    case 'not_addressed':
      return h('div', { class: 'cell' }, label, h('div', { class: 'note' }, 'Not discussed in this call.'));
    case 'missing':
      return h('div', { class: 'cell' }, label,
        h('div', { class: 'note warn' }, 'The model returned no answer for this question. Re-run the analysis.'));
    case 'unverified':
      return h('div', { class: 'cell' }, label,
        h('div', { class: 'note warn' }, 'No quote from the transcript could be verified for this answer. Treat it as unconfirmed.'),
        h('p', { class: 'answer muted' }, a.answer));
    default:
      return h('div', { class: 'cell' }, label,
        a.coverage === 'partial' ? h('div', { class: 'note' }, 'Partly addressed.') : null,
        h('p', { class: 'answer' }, a.answer),
        a.evidence.map((ev) => quoteFigure(ev)));
  }
}

function renderGuide() {
  const root = $('#panel-guide');
  root.replaceChildren();
  if (!state.corpus) return;
  if (!state.analysis) { root.append(emptyState()); return; }
  const experts = state.corpus.transcripts;
  root.style.setProperty('--cols', experts.length);
  root.append(h('div', { class: 'expert-heads' }, experts.map((t) =>
    h('div', { class: 'expert-head' }, h('strong', {}, t.expert_name), h('span', {}, roleMarket(t))))));
  for (const q of state.corpus.guide.questions) {
    root.append(h('section', { class: 'q' },
      h('h3', {}, `${q.id}. ${q.text}`),
      h('div', { class: 'q-row' }, experts.map((t) => guideCell(t, answerFor(t.id, q.id))))));
  }
}

function positionRow(p) {
  const t = state.transcripts.get(p.transcript_id);
  return h('div', { class: 'position' },
    h('div', { class: 'who' }, h('strong', {}, t ? t.expert_name : p.transcript_id), h('span', {}, t ? roleMarket(t) : '')),
    h('div', {}, h('p', { class: 'claim' }, p.position), p.evidence.map((ev) => quoteFigure(ev))));
}

function renderThemes() {
  const root = $('#panel-themes');
  root.replaceChildren();
  if (!state.corpus) return;
  if (!state.analysis) { root.append(emptyState()); return; }
  const { common_themes: themes, disagreements } = state.analysis.themes;

  root.append(h('div', { class: 'group' },
    h('h2', { class: 'group-title agree' }, 'Where the experts agree'),
    themes.length ? themes.map((t) => h('article', { class: 'theme' },
      h('h3', {}, t.title), h('p', { class: 'summary' }, t.summary),
      h('div', { class: 'positions' }, t.positions.map(positionRow))))
      : h('p', { class: 'sub' }, 'No common theme was supported by verified quotes from at least two experts.')));

  root.append(h('div', { class: 'group' },
    h('h2', { class: 'group-title differ' }, 'Where they differ'),
    disagreements.length ? disagreements.map((d) => h('article', { class: 'theme' },
      h('h3', {}, d.topic),
      h('p', { class: 'strength' }, d.strength === 'clear' ? 'Direct conflict' : 'Difference of degree or emphasis'),
      h('p', { class: 'summary' }, d.summary),
      d.nuance ? h('p', { class: 'nuance' }, h('strong', {}, 'Comparability. '), d.nuance) : null,
      h('div', { class: 'positions' }, d.positions.map(positionRow))))
      : h('p', { class: 'sub' }, 'No disagreement was supported by verified quotes from at least two experts.')));
}

function answerNodes(res) {
  const byN = new Map(res.citations.map((c) => [c.n, c]));
  return res.answer.split(/\n{2,}/).map((paragraph) => {
    const nodes = [];
    const marker = /\[(\d+|\?)\]/g;
    let last = 0;
    let m;
    while ((m = marker.exec(paragraph)) !== null) {
      nodes.push(paragraph.slice(last, m.index));
      const cite = m[1] === '?' ? null : byN.get(Number(m[1]));
      if (cite) {
        nodes.push(h('button', {
          class: 'cite', type: 'button',
          'aria-label': `Source ${cite.n}: ${cite.expert_name} at ${cite.timestamp}`,
          onclick: (e) => openSource(cite, e.currentTarget),
        }, String(cite.n)));
      } else {
        nodes.push(h('span', { class: 'cite bad', title: 'This citation could not be verified against the transcripts.' }, '?'));
      }
      last = marker.lastIndex;
    }
    nodes.push(paragraph.slice(last));
    return h('p', {}, nodes);
  });
}

function starterQuestions() {
  const qs = state.corpus ? state.corpus.guide.questions : [];
  if (qs.length <= 3) return qs.map((q) => q.text);
  return [0, 1, 2].map((i) => qs[Math.floor((i * qs.length) / 3)].text);
}

function renderChat() {
  const log = $('#chat-log');
  log.replaceChildren();
  if (!state.corpus) return;
  if (state.chat.length === 0) {
    log.append(h('div', { class: 'starters' },
      h('p', {}, 'Ask across all the calls. Try one of these, or write your own:'),
      starterQuestions().map((text) => h('button', { type: 'button', onclick: () => ask(text) }, text))));
    return;
  }
  for (const item of state.chat) {
    const reply = h('div', { class: 'reply' });
    if (item.pending) {
      reply.append(h('p', { class: 'pending' }, 'Reading the transcripts\u2026'));
    } else if (item.error) {
      reply.append(h('div', { class: 'notice' }, item.error));
    } else {
      const r = item.res;
      if (r.status === 'not_found') reply.append(h('div', { class: 'note' }, 'The transcripts do not cover this.'));
      if (r.status === 'unverified') reply.append(h('div', { class: 'note warn' }, 'No citation could be verified. Treat this answer as unconfirmed.'));
      reply.append(...answerNodes(r));
      if (r.citations.length) {
        reply.append(h('ol', { class: 'sources', 'aria-label': 'Sources' }, r.citations.map((c) =>
          h('li', {}, h('span', { class: 'n' }, String(c.n)), quoteFigure(c, { who: true })))));
      }
      if (r.warnings.length) {
        reply.append(h('div', { class: 'sub' }, `${r.warnings.length} citation ${r.warnings.length === 1 ? 'check' : 'checks'} flagged: ${r.warnings[0]}`));
      }
    }
    log.append(h('div', { class: 'exchange' }, h('div', { class: 'asked' }, item.q), reply));
  }
}

function renderTranscripts() {
  const root = $('#transcript-view');
  root.replaceChildren();
  if (!state.corpus) return;
  const t = state.transcripts.get(state.transcriptId);
  if (!t) return;
  root.append(
    h('div', { class: 'picker' }, state.corpus.transcripts.map((x) =>
      h('button', {
        type: 'button', 'aria-pressed': String(x.id === t.id),
        onclick: () => { state.transcriptId = x.id; renderTranscripts(); },
      }, `${x.expert_name} (${x.market || x.id})`))),
    h('p', { class: 'sub' }, `${roleMarket(t)} \u00B7 ${t.filename}`),
    t.warnings.length ? h('div', { class: 'note warn' }, t.warnings.join(' ')) : null,
    h('div', { class: 'turns' }, t.segments.map((s) => {
      const focused = state.focus && state.focus.segment_id === s.id;
      return h('article', { class: `turn ${s.is_expert ? 'expert' : 'interviewer'}${focused ? ' target' : ''}`, id: `seg-${s.id}` },
        h('div', { class: 't' }, s.timestamp),
        h('div', {}, h('div', { class: 'speaker' }, s.speaker),
          h('p', {}, focused ? markedText(s.text, state.focus.start, state.focus.end) : s.text)));
    })));
}

// ---------------------------------------------------------- source drawer ---
let drawerTrigger = null;

function openSource(ev, trigger) {
  const seg = state.segments.get(ev.segment_id);
  const t = seg && state.transcripts.get(seg.transcript_id);
  if (!seg || !t) return;
  drawerTrigger = trigger || document.activeElement;
  state.drawerEvidence = ev;
  $('#drawer-title').textContent = t.expert_name;
  $('#drawer-sub').textContent = roleMarket(t);
  $('#drawer-body').replaceChildren(...t.segments
    .filter((s) => Math.abs(s.index - seg.index) <= 1)
    .map((s) => h('div', { class: `dturn${s.id === seg.id ? ' current' : ''}` },
      h('div', { class: 'meta' }, h('b', {}, s.timestamp), ` ${s.speaker}`),
      h('p', {}, s.id === seg.id ? markedText(s.text, ev.start, ev.end) : s.text))));
  const drawer = $('#drawer');
  drawer.classList.add('open');
  drawer.setAttribute('aria-hidden', 'false');
  $('#drawer-close').focus();
}

function closeDrawer(restoreFocus = true) {
  const drawer = $('#drawer');
  drawer.classList.remove('open');
  drawer.setAttribute('aria-hidden', 'true');
  if (restoreFocus && drawerTrigger && document.contains(drawerTrigger)) drawerTrigger.focus();
  drawerTrigger = null;
}

function openInTranscript() {
  const ev = state.drawerEvidence;
  const seg = ev && state.segments.get(ev.segment_id);
  if (!seg) return;
  state.transcriptId = seg.transcript_id;
  state.focus = { segment_id: seg.id, start: ev.start, end: ev.end };
  closeDrawer(false);
  setTab('transcripts');
  const target = document.getElementById(`seg-${seg.id}`);
  if (target) target.scrollIntoView({ block: 'center' });
}

// ---------------------------------------------------------------- actions ---
function render() {
  renderHeader();
  renderNotice();
  renderStatus();
  renderGuide();
  renderThemes();
  renderChat();
  renderTranscripts();
  for (const tab of document.querySelectorAll('.tab')) {
    const selected = tab.dataset.tab === state.tab;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
  }
  for (const name of ['guide', 'themes', 'ask', 'transcripts']) $(`#panel-${name}`).hidden = name !== state.tab;
}

function setTab(name) {
  state.tab = name;
  render();
}

async function withBusy(work) {
  state.busy = true;
  state.error = null;
  render();
  try {
    await work();
  } catch (error) {
    state.error = error.message;
  } finally {
    state.busy = false;
    render();
  }
}

const runAnalysis = (refresh) =>
  withBusy(async () => { state.analysis = await postJson('/api/analysis', { refresh }); });

async function afterCorpusChange(corpus) {
  setCorpus(corpus);
  state.analysis = await api('/api/analysis');
  state.chat = [];
  state.focus = null;
}

async function uploadFiles() {
  const files = $('#file-transcripts').files;
  if (!files.length) {
    state.error = 'Choose at least one transcript (.txt) file first.';
    render();
    return;
  }
  const form = new FormData();
  for (const file of files) form.append('transcripts', file);
  const guide = $('#file-guide').files[0];
  if (guide) form.append('guide', guide);
  await withBusy(async () => {
    await afterCorpusChange(await api('/api/corpus', { method: 'POST', body: form }));
    $('#file-transcripts').value = '';
    $('#file-guide').value = '';
  });
}

const useSample = () => withBusy(async () => { await afterCorpusChange(await api('/api/corpus/sample', { method: 'POST' })); });

async function ask(text) {
  const question = text.trim();
  if (!question || state.chat.some((c) => c.pending)) return;
  // Only completed exchanges become history; keep the last five (the API allows ten turns).
  const history = state.chat.filter((c) => c.res).slice(-5)
    .flatMap((c) => [{ role: 'user', content: c.q }, { role: 'assistant', content: c.res.answer }]);
  const item = { q: question, pending: true };
  state.chat.push(item);
  state.error = null;
  $('#question').value = '';
  renderNotice();
  renderChat();
  try {
    item.res = await postJson('/api/ask', { question, history });
  } catch (error) {
    item.error = error.message;
  }
  item.pending = false;
  renderChat();
}

// ------------------------------------------------------------------- init ---
function bindEvents() {
  $('#analyze-btn').addEventListener('click', () => runAnalysis(Boolean(state.analysis)));
  $('#ask-btn').addEventListener('click', () => ask($('#question').value));
  $('#question').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(e.currentTarget.value); }
  });
  $('#upload-btn').addEventListener('click', uploadFiles);
  $('#sample-btn').addEventListener('click', useSample);
  $('#drawer-close').addEventListener('click', () => closeDrawer());
  $('#drawer-open').addEventListener('click', openInTranscript);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && $('#drawer').classList.contains('open')) closeDrawer();
  });
  const tabs = [...document.querySelectorAll('.tab')];
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => setTab(tab.dataset.tab));
    tab.addEventListener('keydown', (e) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (!step) return;
      const next = tabs[(i + step + tabs.length) % tabs.length];
      setTab(next.dataset.tab);
      next.focus();
    });
  });
}

async function init() {
  bindEvents();
  try {
    const [health, corpus, analysis] = await Promise.all([api('/api/health'), api('/api/corpus'), api('/api/analysis')]);
    state.health = health;
    setCorpus(corpus);
    state.analysis = analysis;
  } catch (error) {
    state.error = `Could not load the app: ${error.message}`;
  }
  render();
}

init();
