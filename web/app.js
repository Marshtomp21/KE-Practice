/**
 * 页面逻辑：提问 -> 渲染答案 / 引用 / 子图。
 *
 * 三条硬性要求在这里落地：
 * 1. 答案里出现的实体可点击，点击后定位到右栏节点并展示其证据；
 * 2. 引用原文按命中区间高亮，高亮依据是本次子图里的关键实体名称；
 * 3. 空结果、超时、后端异常都给出明确提示，界面不留白屏。
 *
 * 答案渲染做了结构化：结构化生成器输出的「【关系】共 N 条：A → B；…」按分组卡片渲染，
 * 「结论：…」渲染成醒目的判定条。模型生成的自由文本走段落分支，两者共用同一套实体
 * 链接与引用编号样式，所以换生成器不需要改前端。
 */
const RETRIEVER_META = {
  vector: { title: '向量 RAG', hint: '使用本地向量索引召回文本片段，再由统一生成器回答。' },
  library_graphrag: { title: '库 GraphRAG', hint: '调用 neo4j-graphrag：向量命中 Neo4j Chunk 后扩展有界实体邻域，再由 GraphRAG 生成。' },
};

const dom = {
  form: document.getElementById('askForm'),
  question: document.getElementById('questionInput'),
  counter: document.getElementById('charCounter'),
  retrieverGroup: document.getElementById('retrieverGroup'),
  hint: document.getElementById('retrieverHint'),
  topK: document.getElementById('topKInput'),
  topKValue: document.getElementById('topKValue'),
  button: document.getElementById('askButton'),
  exampleBlock: document.getElementById('exampleBlock'),
  exampleList: document.getElementById('exampleList'),
  history: document.getElementById('historyList'),
  clearHistory: document.getElementById('clearHistory'),
  answer: document.getElementById('answerBox'),
  answerMeta: document.getElementById('answerMeta'),
  citations: document.getElementById('citationBox'),
  citationCount: document.getElementById('citationCount'),
  graphCanvas: document.getElementById('graphCanvas'),
  graphEmpty: document.getElementById('graphEmpty'),
  graphMeta: document.getElementById('graphMeta'),
  graphLegend: document.getElementById('graphLegend'),
  resetView: document.getElementById('resetView'),
  zoomControls: document.getElementById('zoomControls'),
  zoomIn: document.getElementById('zoomIn'),
  zoomOut: document.getElementById('zoomOut'),
  zoomFit: document.getElementById('zoomFit'),
  nodeDetail: document.getElementById('nodeDetail'),
  stats: document.getElementById('graphStats'),
  toast: document.getElementById('toast'),
};

const state = {
  history: [],
  retriever: null,
  lastGraph: { nodes: [], edges: [] },
  graphHandle: null,
};

/* ---------- 工具 ---------- */

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

const escapeRegExp = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function toast(message) {
  dom.toast.textContent = message;
  dom.toast.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { dom.toast.hidden = true; }, 5600);
}

async function request(path, options) {
  const controller = new AbortController();
  // 后端模型超时为 120 秒，浏览器需留出检索、网络和序列化余量。
  const timeoutMs = 150000;
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `后端返回 ${response.status}`);
    return body;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error(`请求超时（${timeoutMs / 1000} 秒），后端可能仍在检索`);
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

/* ---------- 启动 ---------- */

async function loadHealth() {
  try {
    const health = await request('/api/health');
    if (!health.ready) {
      dom.stats.innerHTML = `<span class="chip chip-error">${escapeHtml(health.detail || '后端未就绪')}</span>`;
      toast(health.detail || '后端未就绪，请先运行 python scripts/build_index.py');
      return;
    }

    state.retriever = health.default_retriever;
    dom.retrieverGroup.innerHTML = health.retrievers
      .map((name) => {
        const meta = RETRIEVER_META[name] || { title: name };
        return `<button type="button" class="seg" role="radio" data-name="${name}"
          aria-checked="${name === state.retriever}">${escapeHtml(meta.title)}</button>`;
      })
      .join('');
    dom.retrieverGroup.querySelectorAll('.seg').forEach((button) => {
      button.addEventListener('click', () => selectRetriever(button.dataset.name));
    });
    updateHint();

    const resources = health.graph || {};
    dom.stats.innerHTML = [
      ['本地向量索引', resources.local_vector_index ? '就绪' : '未构建'],
      ['库 GraphRAG', 'Neo4j'],
    ].map(([label, value]) => `<span class="chip">${label} <b>${value}</b></span>`).join('');
  } catch (error) {
    dom.stats.innerHTML = '<span class="chip chip-error">无法连接后端</span>';
    toast(error.message);
  }
}

async function loadExamples() {
  try {
    const body = await request('/api/examples');
    const examples = body.examples || [];
    if (!examples.length) return;
    dom.exampleBlock.hidden = false;
    dom.exampleList.innerHTML = examples
      .map((item) => `<button type="button" class="example" data-q="${escapeHtml(item.question)}">
        <span class="kind">${escapeHtml(item.label)}</span>
        <span>${escapeHtml(item.question)}</span>
      </button>`)
      .join('');
    dom.exampleList.querySelectorAll('.example').forEach((button) => {
      button.addEventListener('click', () => {
        dom.question.value = button.dataset.q;
        updateCounter();
        dom.form.requestSubmit();
      });
    });
  } catch (error) {
    // 示例只是锦上添花，取不到就安静跳过，不打扰用户
  }
}

function selectRetriever(name) {
  state.retriever = name;
  dom.retrieverGroup.querySelectorAll('.seg').forEach((button) => {
    button.setAttribute('aria-checked', String(button.dataset.name === name));
  });
  updateHint();
}

function updateHint() {
  const meta = RETRIEVER_META[state.retriever];
  dom.hint.textContent = meta ? meta.hint : '';
}

function updateCounter() {
  dom.counter.textContent = String(dom.question.value.length);
}

/* ---------- 答案渲染 ---------- */

/**
 * 把一批名称在文本里替换成标记。
 *
 * 必须一次扫描完成：如果逐个名称反复 replace，先插入的 HTML 属性
 * （如 data-entity="石宁安"）会被后面较短的名称再次命中，产出嵌套坏标签。
 * 因此把所有名称合成一条按长度降序的交替正则，长名优先匹配，只走一遍。
 */
function markNames(text, names, wrap) {
  const ordered = [...new Set(names)]
    .filter((name) => name && name.length >= 2)
    .sort((a, b) => b.length - a.length);
  const escaped = escapeHtml(text);
  if (!ordered.length) return escaped;
  const pattern = new RegExp(ordered.map((n) => escapeRegExp(escapeHtml(n))).join('|'), 'g');
  return escaped.replace(pattern, (hit) => wrap(hit));
}

/** 把答案里出现的实体名做成可点击锚点，并把 [S1] / [G1] 编号渲染成角标。 */
function linkEntities(text, nodes) {
  const html = markNames(
    text,
    nodes.map((n) => n.label),
    (name) => `<span class="entity" data-entity="${name}">${name}</span>`,
  );
  return html.replace(/\[(S\d+|G\d+)\]/g, '<span class="marker">$1</span>');
}

/** 「A → B（1998）[S1]」拆成三段渲染，年份与引用编号各自成样式。 */
function renderRelationItem(raw, nodes) {
  const item = raw.trim();
  if (!item) return '';
  const parts = item.split('→');
  if (parts.length !== 2) return `<span class="rel-item">${linkEntities(item, nodes)}</span>`;
  const head = parts[0].trim();
  const rest = parts[1].trim();
  const yearMatch = rest.match(/（(\d{4})）/);
  const tail = rest.replace(/（\d{4}）/, '').replace(/\[[SG]\d+\]/g, '').trim();
  const markers = (rest.match(/\[[SG]\d+\]/g) || []).join('');
  return `<span class="rel-item">${linkEntities(head, nodes)}<span class="arrow">→</span>${linkEntities(tail, nodes)}`
    + (yearMatch ? `<span class="year">${yearMatch[1]}</span>` : '')
    + (markers ? linkEntities(markers, []) : '')
    + '</span>';
}

function renderAnswer(text, nodes) {
  const lines = String(text).split('\n').map((line) => line.trim()).filter(Boolean);
  const blocks = [];

  lines.forEach((line) => {
    // 引用原文在下方「引用原文」区已完整列出，答案正文里不再重复
    if (/^\[S\d+\]\s*原文/.test(line)) return;

    const group = line.match(/^【(.+?)】共\s*(\d+)\s*条[：:]([\s\S]*)$/);
    if (group) {
      const [, label, count, body] = group;
      const trailing = body.match(/，另有\s*(\d+)\s*条同类关系。?$/);
      const items = body
        .replace(/，另有\s*\d+\s*条同类关系。?$/, '')
        .replace(/。$/, '')
        .split('；')
        .map((piece) => renderRelationItem(piece, nodes))
        .filter(Boolean)
        .join('');
      blocks.push(`<div class="rel-group">
        <div class="rel-head"><span class="rel-name">${escapeHtml(label)}</span>
          <span class="rel-count">${count} 条</span></div>
        <div class="rel-items">${items}${trailing ? `<span class="rel-more">另有 ${trailing[1]} 条同类关系</span>` : ''}</div>
      </div>`);
      return;
    }

    if (/^结论[：:]/.test(line)) {
      const body = line.replace(/^结论[：:]\s*/, '');
      // 「不存在直接关系，但通过 X 产生关联」是一个肯定结论，不能按否定着色；
      // 只有彻底查无此关系时才标红。
      const connected = /产生关联|存在「|存在“/.test(body);
      const denied = /不存在|查不到|没有|未能|无法/.test(body);
      const tone = denied && !connected ? 'negative' : 'positive';
      blocks.push(`<div class="verdict ${tone}">
        <span class="verdict-tag">结论</span>
        <span>${linkEntities(body, nodes)}</span>
      </div>`);
      return;
    }

    if (/^本次检索|^根据现有语料/.test(line)) {
      blocks.push(`<div class="notice">${linkEntities(line, nodes)}</div>`);
      return;
    }

    blocks.push(`<p class="para">${linkEntities(line, nodes)}</p>`);
  });

  return `<div class="answer-body fade-in">${blocks.join('')}</div>`;
}

/** 引用原文里把子图的关键实体高亮出来，作为「命中区间」的可视提示。 */
function highlightSnippet(snippet, nodes) {
  return markNames(
    snippet,
    nodes.filter((n) => n.highlight).map((n) => n.label),
    (name) => `<mark>${name}</mark>`,
  );
}

/* ---------- 子图与详情 ---------- */

function focusEntity(name) {
  const node = state.lastGraph.nodes.find((n) => n.label === name);
  if (!node) {
    toast(`本次子图里没有「${name}」这个节点`);
    return;
  }
  showNodeDetail(node);
  dom.nodeDetail.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showNodeDetail(node) {
  const evidences = node.evidences || [];
  const body = evidences.length
    ? evidences.map((e) => `<div class="ev">
        <div class="where">${escapeHtml(e.doc_id)} [${e.char_start}:${e.char_end}] · 置信度 ${e.confidence}</div>
        <div class="quote">${escapeHtml(String(e.raw_text).slice(0, 110))}</div>
      </div>`).join('')
    : '<div class="ev"><div class="quote">该节点没有附带证据记录。</div></div>';
  const aliases = (node.aliases || []).length
    ? `<div class="ev"><div class="where">别名</div><div class="quote">${escapeHtml(node.aliases.join('、'))}</div></div>`
    : '';

  dom.nodeDetail.innerHTML = `<div class="fade-in">
    <div class="detail-head">
      <span class="dot" style="width:9px;height:9px;border-radius:50%;background:${GraphView.colorOf(node.type)}"></span>
      <span class="name">${escapeHtml(node.label)}</span>
      <span class="pill pill-muted">${escapeHtml(node.type_label)}</span>
      <span class="score">得分 ${node.score}</span>
    </div>
    <div class="detail-body">${aliases}${body}</div>
  </div>`;
}

/* ---------- 主流程 ---------- */

function renderHistory() {
  dom.clearHistory.hidden = !state.history.length;
  if (!state.history.length) {
    dom.history.innerHTML = '<li class="empty">还没有提问记录</li>';
    return;
  }
  dom.history.innerHTML = state.history
    .map((item, index) => `<li class="item" data-index="${index}">
      <div class="q">${escapeHtml(item.question)}</div>
      <div class="tag"><span>${escapeHtml(item.retriever)}</span><span>${item.latency}s</span></div>
    </li>`)
    .join('');
  dom.history.querySelectorAll('li[data-index]').forEach((node) => {
    node.addEventListener('click', () => {
      const item = state.history[Number(node.dataset.index)];
      dom.question.value = item.question;
      updateCounter();
      selectRetriever(item.retriever);
      showResult(item.payload);
    });
  });
}

function showResult(payload) {
  const nodes = (payload.graph && payload.graph.nodes) || [];
  const edges = (payload.graph && payload.graph.edges) || [];
  state.lastGraph = { nodes, edges };

  const retrieved = (payload.debug && payload.debug.retrieved) || {};
  dom.answerMeta.innerHTML = [
    `<span class="pill pill-accent">${escapeHtml(payload.retriever)}</span>`,
    `<span class="count-badge">${payload.latency}s</span>`,
    retrieved.relations !== undefined
      ? `<span class="count-badge">关系 ${retrieved.relations}</span>` : '',
  ].join('');

  dom.answer.innerHTML = renderAnswer(payload.answer, nodes);
  dom.answer.querySelectorAll('.entity').forEach((node) => {
    node.addEventListener('click', () => focusEntity(node.dataset.entity));
  });

  const citations = payload.citations || [];
  dom.citationCount.hidden = !citations.length;
  dom.citationCount.textContent = `${citations.length} 段`;
  dom.citations.innerHTML = citations.length
    ? citations.map((c) => `<div class="citation fade-in">
        <div class="meta">
          <span class="pill pill-muted">${escapeHtml(c.marker)}</span>
          <span class="src">${escapeHtml(c.doc_id)}</span>
          <span>字符区间 [${c.char_start}:${c.char_end}]</span>
        </div>
        <div class="text">${highlightSnippet(c.snippet, nodes)}</div>
      </div>`).join('')
    : '<div class="placeholder small"><p>本次检索没有命中任何原文片段。</p></div>';

  dom.graphMeta.hidden = !nodes.length;
  dom.graphMeta.textContent = `节点 ${nodes.length} · 边 ${edges.length}`;
  dom.resetView.hidden = !nodes.length;
  dom.zoomControls.hidden = !nodes.length;
  dom.graphEmpty.hidden = nodes.length > 0;

  state.graphHandle = GraphView.render(dom.graphCanvas, payload.graph || {}, {
    onSelect: showNodeDetail,
  });
  dom.graphLegend.innerHTML = (state.graphHandle.types || [])
    .map((t) => `<span class="lg"><span class="dot" style="background:${t.color}"></span>${escapeHtml(t.label)} ${t.count}</span>`)
    .join('');

  if (!nodes.length) {
    dom.nodeDetail.innerHTML = '<div class="placeholder small"><p>本次检索没有返回子图（向量基线不查图，属正常）。</p></div>';
  } else {
    dom.nodeDetail.innerHTML = '<div class="placeholder small"><p>点击节点可查看它的证据来源。</p></div>';
  }
}

dom.form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = dom.question.value.trim();
  if (!question) {
    toast('请先输入问题');
    dom.question.focus();
    return;
  }

  dom.button.disabled = true;
  dom.button.classList.add('busy');
  dom.button.querySelector('.label').textContent = '检索中…';
  try {
    const payload = await request('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        retriever: state.retriever || null,
        top_k: Number(dom.topK.value) || null,
      }),
    });
    showResult(payload);
    state.history.unshift({
      question,
      retriever: payload.retriever,
      latency: payload.latency,
      payload,
    });
    state.history = state.history.slice(0, 20);
    renderHistory();
  } catch (error) {
    toast(error.message);
    dom.answer.innerHTML = `<div class="placeholder"><p>请求失败</p><p class="sub">${escapeHtml(error.message)}</p></div>`;
  } finally {
    dom.button.disabled = false;
    dom.button.classList.remove('busy');
    dom.button.querySelector('.label').textContent = '提问';
  }
});

// Ctrl/Cmd + Enter 直接提交，演示时不必去够鼠标
dom.question.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') dom.form.requestSubmit();
});
dom.question.addEventListener('input', updateCounter);
dom.topK.addEventListener('input', () => { dom.topKValue.textContent = dom.topK.value; });
dom.resetView.addEventListener('click', () => state.graphHandle && state.graphHandle.reset());
dom.zoomIn.addEventListener('click', () => state.graphHandle && state.graphHandle.zoomBy(1.3));
dom.zoomOut.addEventListener('click', () => state.graphHandle && state.graphHandle.zoomBy(1 / 1.3));
dom.zoomFit.addEventListener('click', () => state.graphHandle && state.graphHandle.fit());
dom.clearHistory.addEventListener('click', () => {
  state.history = [];
  renderHistory();
});

// 改窗口大小只需重新适配画布，不必重跑布局——重跑会把节点位置全部打乱
let resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => state.graphHandle && state.graphHandle.fit(), 180);
});

// 系统主题切换时子图配色要跟着走：Cytoscape 画在 canvas 上，读不到 CSS 变量，
// 只能整张重绘一次
const themeQuery = window.matchMedia('(prefers-color-scheme: dark)');
const onThemeChange = () => {
  if (state.lastGraph.nodes.length) {
    state.graphHandle = GraphView.render(dom.graphCanvas, state.lastGraph, { onSelect: showNodeDetail });
  }
};
if (themeQuery.addEventListener) themeQuery.addEventListener('change', onThemeChange);

loadHealth();
loadExamples();
updateCounter();
