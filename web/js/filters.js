// One-Euro filter (Casiez et al. 2012) with scale-normalised speed, so the same
// parameters work for a small far-away hand and a big close-up face.
const TAU = Math.PI * 2;

function alpha(cutoff, dt) {
  const r = TAU * cutoff * dt;
  return r / (r + 1);
}

export class OneEuro {
  constructor(minCutoff = 1.0, beta = 1.0, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.x = null;
    this.dx = 0;
  }
  reset() { this.x = null; this.dx = 0; }
  filter(value, dt, scale = 1) {
    if (this.x === null || !(dt > 0)) { this.x = value; this.dx = 0; return value; }
    const d = (value - this.x) / dt;
    this.dx += alpha(this.dCutoff, dt) * (d - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx) / Math.max(scale, 1e-6);
    this.x += alpha(cutoff, dt) * (value - this.x);
    return this.x;
  }
}

// Filters a flat Float32Array of N points with `dim` components each.
export class PointSetFilter {
  constructor(n, dim, minCutoff, beta, dCutoff = 1.0) {
    this.n = n; this.dim = dim;
    this.f = Array.from({ length: n * dim }, () => new OneEuro(minCutoff, beta, dCutoff));
    this.out = new Float32Array(n * dim);
  }
  reset() { for (const f of this.f) f.reset(); }
  apply(src, dt, scale, components = this.dim) {
    const { n, dim, f, out } = this;
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < dim; k++) {
        const j = i * dim + k;
        out[j] = k < components ? f[j].filter(src[j], dt, scale) : src[j];
      }
    }
    return out;
  }
}
