/**
 * 子图渲染：Cytoscape.js + 力导向布局（cose）。
 *
 * 早期版本是手写的弹簧-斥力迭代，跑固定轮数就停，节点分布与边的走向都不好看。
 * 换成成熟图库后由库负责布局收敛与 canvas 渲染，我们只负责三件事：
 * 把后端返回的 { nodes, edges } 映射成图元、按主题令牌配色、接上交互。
 *
 * 视觉约定：
 * - 节点直径映射 RetrievalResult.scores，得分越高越大；
 * - 本次推理路径上的节点加描边、边用强调色并常显关系标签；
 * - 悬停节点时聚焦其一跳邻域，其余元素淡出。
 *
 * 库文件随仓库分发（web/vendor/cytoscape.min.js），不依赖 CDN，断网也能演示。
 */
const GraphView = (() => {
  const TYPE_COLOR = {
    Person: '#2d6cdf',
    Movie: '#0f9d8b',
    Company: '#b45309',
    Award: '#8b46d6',
    Genre: '#64748b',
    Character: '#d1435b',
  };
  const FALLBACK_COLOR = '#7a8595';
  const colorOf = (type) => TYPE_COLOR[type] || FALLBACK_COLOR;

  /** Cytoscape 画在 canvas 上，读不到 CSS 变量，渲染前先把主题令牌取成实际色值。 */
  function themeTokens() {
    const style = getComputedStyle(document.documentElement);
    const read = (name, fallback) => (style.getPropertyValue(name) || '').trim() || fallback;
    return {
      ink: read('--ink', '#131a26'),
      inkSoft: read('--ink-soft', '#5b6779'),
      inkFaint: read('--ink-faint', '#8b95a6'),
      line: read('--line-strong', '#cfd7e3'),
      accent: read('--accent', '#2d6cdf'),
      surface: read('--surface-soft', '#f8fafc'),
    };
  }

  const sizeOf = (score) => 16 + Math.sqrt(Math.max(Number(score) || 0, 0)) * 34;

  function buildStyle(theme) {
    return [
      {
        selector: 'node',
        style: {
          'background-color': 'data(color)',
          'background-opacity': 0.85,
          width: 'data(size)',
          height: 'data(size)',
          label: 'data(label)',
          color: theme.ink,
          'font-size': 10,
          'font-family': 'inherit',
          'text-valign': 'bottom',
          'text-margin-y': 4,
          'text-outline-color': theme.surface,
          'text-outline-width': 2.5,
          'text-max-width': 96,
          'text-wrap': 'ellipsis',
          'border-width': 1.5,
          'border-color': theme.surface,
          'transition-property': 'opacity, border-width, background-opacity',
          'transition-duration': '140ms',
        },
      },
      {
        // 锚点与高分节点：加粗描边并让标签更醒目
        selector: 'node[?highlight]',
        style: {
          'background-opacity': 1,
          'border-width': 3,
          'border-color': 'data(color)',
          'border-opacity': 0.45,
          'font-size': 11,
          'font-weight': 'bold',
        },
      },
      {
        selector: 'edge',
        style: {
          width: 1.1,
          'line-color': theme.line,
          'target-arrow-color': theme.line,
          'target-arrow-shape': 'triangle',
          'arrow-scale': 0.75,
          'curve-style': 'bezier',
          'control-point-step-size': 26,
          'transition-property': 'opacity, line-color, width',
          'transition-duration': '140ms',
        },
      },
      {
        selector: 'edge[?highlight]',
        style: {
          width: 2,
          'line-color': theme.accent,
          'target-arrow-color': theme.accent,
        },
      },
      {
        // 边标签默认不显示：上百条边同时标注会糊成一片，只在悬停与高亮时给出
        selector: 'edge.labelled',
        style: {
          label: 'data(label)',
          'font-size': 9,
          color: theme.inkSoft,
          'text-outline-color': theme.surface,
          'text-outline-width': 2.5,
          'text-rotation': 'autorotate',
        },
      },
      { selector: '.faded', style: { opacity: 0.12, 'text-opacity': 0 } },
      { selector: '.dim-label', style: { 'text-opacity': 0.25 } },
    ];
  }

  const LAYOUT = {
    name: 'cose',
    animate: false,
    randomize: false,
    componentSpacing: 70,
    nodeRepulsion: () => 9000,
    idealEdgeLength: () => 78,
    edgeElasticity: () => 90,
    nestingFactor: 1.1,
    gravity: 55,
    numIter: 1400,
    initialTemp: 220,
    coolingFactor: 0.95,
    minTemp: 1.0,
    nodeDimensionsIncludeLabels: true,
    fit: true,
    padding: 26,
  };

  let instance = null;

  function render(container, payload, handlers) {
    const onSelect = (handlers && handlers.onSelect) || (() => {});
    const nodes = payload.nodes || [];
    const edges = payload.edges || [];

    if (instance) {
      instance.destroy();
      instance = null;
    }
    if (!nodes.length) return { types: [], reset: () => {}, fit: () => {} };

    const known = new Set(nodes.map((n) => n.id));
    const theme = themeTokens();

    const elements = [
      ...nodes.map((node) => ({
        data: {
          id: node.id,
          label: node.label,
          type: node.type,
          typeLabel: node.type_label,
          color: colorOf(node.type),
          size: sizeOf(node.score),
          score: node.score,
          highlight: node.highlight ? 1 : 0,
          payload: node,
        },
      })),
      ...edges
        .filter((edge) => known.has(edge.source) && known.has(edge.target))
        .map((edge) => ({
          data: {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            label: edge.start_year ? `${edge.label} ${edge.start_year}` : edge.label,
            highlight: edge.highlight ? 1 : 0,
          },
        })),
    ];

    instance = cytoscape({
      container,
      elements,
      style: buildStyle(theme),
      layout: LAYOUT,
      wheelSensitivity: 0.22,
      minZoom: 0.2,
      maxZoom: 3.5,
      pixelRatio: window.devicePixelRatio || 1,
      autoungrabify: false,
    });

    instance.edges('[?highlight]').addClass('labelled');

    /* ---- 悬停聚焦一跳邻域 ---- */
    instance.on('mouseover', 'node', (event) => {
      const focus = event.target.closedNeighborhood();
      instance.elements().difference(focus).addClass('faded');
      focus.edges().addClass('labelled');
    });
    instance.on('mouseout', 'node', () => {
      instance.elements().removeClass('faded');
      instance.edges().not('[?highlight]').removeClass('labelled');
    });
    instance.on('mouseover', 'edge', (event) => event.target.addClass('labelled'));
    instance.on('mouseout', 'edge', (event) => {
      if (!event.target.data('highlight')) event.target.removeClass('labelled');
    });

    instance.on('tap', 'node', (event) => onSelect(event.target.data('payload')));
    // 点空白处取消聚焦，避免拖拽后残留淡出状态
    instance.on('tap', (event) => {
      if (event.target === instance) instance.elements().removeClass('faded');
    });

    const types = [...new Set(nodes.map((n) => n.type))]
      .map((type) => ({
        type,
        label: (nodes.find((n) => n.type === type) || {}).type_label || type,
        color: colorOf(type),
        count: nodes.filter((n) => n.type === type).length,
      }))
      .sort((a, b) => b.count - a.count);

    return {
      types,
      reset() {
        instance.elements().removeClass('faded');
        instance.layout(LAYOUT).run();
        instance.fit(undefined, 26);
      },
      fit() {
        instance.fit(undefined, 26);
      },
      zoomBy(factor) {
        instance.zoom({
          level: Math.min(3.5, Math.max(0.2, instance.zoom() * factor)),
          renderedPosition: { x: container.clientWidth / 2, y: container.clientHeight / 2 },
        });
      },
    };
  }

  return { render, colorOf };
})();
