/* ── STATE ─────────────────────────────────────────────────────────────────── */
const state = {
  currentView: 'home',
  currentPage: 1,
  pageSize: 12,
  totalPages: 1,
};

/* ── VIEW ROUTER ───────────────────────────────────────────────────────────── */
function showView(name) {
  ['home', 'detail', 'stats', 'about'].forEach(v => {
    const el = document.getElementById(`view-${v}`);
    if (el) el.classList.toggle('hidden', v !== name);
  });
  state.currentView = name;
  window.scrollTo({ top: 0, behavior: 'smooth' });

  if (name === 'stats') loadStats();
}

/* ── HERO ──────────────────────────────────────────────────────────────────── */
async function loadHero() {
  try {
    const [statsRes, heroRes] = await Promise.all([
      fetch('/api/stats'),
      fetch('/api/hero'),
    ]);
    const stats   = await statsRes.json();
    const bundles = await heroRes.json();

    // Metrics
    animateCount('hero-bundles',  stats.total_bundles);
    animateCount('hero-products', stats.total_products);
    document.getElementById('hero-avg').textContent = stats.avg_per_bundle;

    // Hero images — pick 4 bundles with valid images
    const wrap = document.getElementById('heroImgWrap');
    const items = bundles.bundles.filter(b => b.image).slice(0, 4);
    wrap.innerHTML = items.map((b, i) => `
      <img src="${b.image}"
           alt="Outfit"
           loading="${i === 0 ? 'eager' : 'lazy'}"
           onerror="this.closest('.hero-img-wrap').removeChild(this)"
      />
    `).join('');
  } catch (e) {
    console.warn('Hero load failed:', e);
  }
}

function animateCount(id, target) {
  const el   = document.getElementById(id);
  if (!el) return;
  const dur  = 1400;
  const step = 16;
  let cur    = 0;
  const inc  = target / (dur / step);
  const t    = setInterval(() => {
    cur += inc;
    if (cur >= target) { cur = target; clearInterval(t); }
    el.textContent = Math.round(cur).toLocaleString();
  }, step);
}

/* ── OUTFIT GRID ───────────────────────────────────────────────────────────── */
async function loadGrid(page = 1) {
  state.currentPage = page;
  const grid  = document.getElementById('outfitGrid');
  const count = document.getElementById('totalCount');

  grid.innerHTML = `
    <div class="loading-wrap">
      <div class="loading-dot"></div>
    </div>`;

  try {
    const res  = await fetch(`/api/bundles?page=${page}&size=${state.pageSize}`);
    const data = await res.json();

    state.totalPages = data.pages;
    count.textContent = `${data.total.toLocaleString()} outfits`;

    grid.innerHTML = data.bundles.map((b, i) => `
      <div class="outfit-card"
           style="animation-delay:${(i % state.pageSize) * 40}ms"
           onclick="loadDetail('${b.id}')">
        ${b.image
          ? `<img src="${b.image}" alt="Outfit" loading="lazy"
                  onerror="this.style.display='none'" />`
          : `<div style="width:100%;height:100%;background:var(--off)"></div>`
        }
        <span class="card-section-badge">${b.section}</span>
        <div class="outfit-card-overlay">
          <div class="outfit-card-info">
            <div class="card-count">${b.n_products} products matched</div>
            <div class="card-id">${b.id}</div>
          </div>
        </div>
      </div>
    `).join('');

    renderPagination(data.page, data.pages);
  } catch (e) {
    grid.innerHTML = `<p style="grid-column:1/-1;padding:40px;color:var(--mid)">Failed to load outfits.</p>`;
  }
}

function renderPagination(current, total) {
  const el = document.getElementById('pagination');
  if (total <= 1) { el.innerHTML = ''; return; }

  let pages = [];
  if (total <= 7) {
    pages = Array.from({ length: total }, (_, i) => i + 1);
  } else {
    pages = [1];
    if (current > 3) pages.push('…');
    for (let p = Math.max(2, current - 1); p <= Math.min(total - 1, current + 1); p++) pages.push(p);
    if (current < total - 2) pages.push('…');
    pages.push(total);
  }

  el.innerHTML = `
    <button onclick="loadGrid(${current - 1})" ${current === 1 ? 'disabled' : ''}>←</button>
    ${pages.map(p =>
      p === '…'
        ? `<button disabled style="border:none;opacity:0.3">…</button>`
        : `<button onclick="loadGrid(${p})" class="${p === current ? 'active' : ''}">${p}</button>`
    ).join('')}
    <button onclick="loadGrid(${current + 1})" ${current === total ? 'disabled' : ''}>→</button>
  `;
}

function scrollToGrid() {
  document.getElementById('gridSection').scrollIntoView({ behavior: 'smooth' });
}

/* ── DETAIL VIEW ───────────────────────────────────────────────────────────── */
async function loadDetail(bundleId) {
  showView('detail');
  const layout = document.getElementById('detailLayout');
  layout.innerHTML = `<div class="loading-wrap"><div class="loading-dot"></div></div>`;

  try {
    const res  = await fetch(`/api/bundle/${bundleId}`);
    const data = await res.json();

    layout.innerHTML = `
      <!-- Left: bundle image -->
      <div class="detail-bundle-img">
        ${data.image
          ? `<img src="${data.image}" alt="Outfit" />`
          : `<div style="aspect-ratio:3/4;background:var(--off)"></div>`
        }
        <div class="detail-bundle-meta">
          <h2>Outfit</h2>
          <p>Section ${data.section} &nbsp;·&nbsp; ${data.predicted.length} products matched</p>
          <p style="margin-top:6px;font-size:10px;color:var(--light);font-family:monospace">
            ${data.id}
          </p>
        </div>
      </div>

      <!-- Right: predicted products -->
      <div class="detail-products">
        <div class="detail-products-header">
          <h3>Matched Products</h3>
          <p>Predicted by MXNJ outfit intelligence model</p>
        </div>
        <div class="products-grid">
          ${data.predicted.map((p, i) => `
            <div class="product-card" style="animation-delay:${i * 35}ms">
              <div class="product-card-img">
                ${p.image
                  ? `<img src="${p.image}" alt="${p.description}" loading="lazy"
                          onerror="this.closest('.product-card-img').style.background='var(--off)';this.style.display='none'" />`
                  : `<div style="width:100%;height:100%;background:var(--off)"></div>`
                }
              </div>
              <div class="product-card-body">
                <div class="product-card-desc">${p.description}</div>
                <div class="product-rank">#${String(i + 1).padStart(2, '0')}</div>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  } catch (e) {
    layout.innerHTML = `<p style="padding:40px;color:var(--mid)">Failed to load bundle.</p>`;
  }
}

/* ── STATS VIEW ────────────────────────────────────────────────────────────── */
async function loadStats() {
  const grid = document.getElementById('statsGrid');
  if (grid.innerHTML.trim()) return; // already loaded

  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();
    const max  = data.top_categories[0]?.count || 1;

    grid.innerHTML = `
      <div class="stat-block" style="animation-delay:0ms">
        <div class="stat-block-label">Outfits analysed</div>
        <div class="stat-block-val">${data.total_bundles.toLocaleString()}</div>
        <div class="stat-block-desc">Unique outfit images processed from the Inditex test set</div>
      </div>

      <div class="stat-block" style="animation-delay:80ms">
        <div class="stat-block-label">Unique products matched</div>
        <div class="stat-block-val">${data.total_products.toLocaleString()}</div>
        <div class="stat-block-desc">Distinct catalogue items identified across all outfits</div>
      </div>

      <div class="stat-block" style="animation-delay:160ms">
        <div class="stat-block-label">Avg. per outfit</div>
        <div class="stat-block-val">${data.avg_per_bundle}</div>
        <div class="stat-block-desc">Mean number of products matched per bundle</div>
      </div>

      <div class="stat-block wide" style="animation-delay:240ms">
        <div class="stat-block-label">Top matched categories</div>
        <div class="category-bars">
          ${data.top_categories.map(c => `
            <div class="cat-bar-row">
              <span class="cat-bar-label">${c.name}</span>
              <div class="cat-bar-track">
                <div class="cat-bar-fill" style="width:0%" data-target="${Math.round(c.count / max * 100)}%"></div>
              </div>
              <span class="cat-bar-count">${c.count}</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;

    // Animate bars after paint
    requestAnimationFrame(() => {
      document.querySelectorAll('.cat-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.target;
      });
    });
  } catch (e) {
    grid.innerHTML = `<p style="padding:40px;color:var(--mid)">Failed to load stats.</p>`;
  }
}

/* ── INIT ──────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  loadHero();
  loadGrid(1);
});