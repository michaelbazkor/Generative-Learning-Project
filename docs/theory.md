# Theory Notes: Curvature Signals for OT Flow Matching

These notes justify the estimators and adaptation laws used in the experiment.
Empirical choices still appear in `REPORT.md`; here we only keep identities that
can be checked without fitting a network.

## 1. Ideal OT paths have zero particle acceleration

Under the linear OT interpolant

$$
x_t = (1-t)\,x_0 + t\,x_1,\qquad t\in[0,1],
$$

the conditional velocity is the constant vector field

$$
v(x_t,t\mid x_0,x_1) = x_1 - x_0.
$$

Differentiating along the characteristic gives

$$
\ddot{x}_t = \frac{d}{dt}v(x_t,t) = 0.
$$

Thus any nonzero measured acceleration on a well-trained OT-coupled model is
necessarily an approximation error or a CFG-induced distortion.

**Independent coupling caveat.** If $x_0$ and $x_1$ are sampled independently,
conditional paths are straight but cross. The *marginal* field
$v_t(x)=\mathbb{E}[x_1-x_0\mid x_t=x]$ is generally curved, so
$\lVert a_t\rVert\not\approx 0$ even for a perfect network. Minibatch OT restores
(approximately) non-crossing straight characteristics.

## 2. Material derivative vs partial time derivative

Along a trajectory $\dot{x} = v(x,t)$,

$$
a_t := \frac{d}{dt}v(x_t,t)
= \partial_t v(x_t,t) + \bigl(\nabla_x v(x_t,t)\bigr)\,v(x_t,t).
$$

The proposal's fixed-$x$ finite difference approximates only $\partial_t v$.
The experiment plan's Heun pair

$$
\tilde{x} = x + \Delta t\,v(x,t),\qquad
\hat{a} = \frac{v(\tilde{x},t+\Delta t)-v(x,t)}{\Delta t}
$$

is a consistent estimator of the *material* derivative:

$$
\hat{a} = a_t + O(\Delta t).
$$

Proof: Taylor-expand $v(\tilde{x},t+\Delta t)$ to first order in $\Delta t$.

## 3. Heun gap equals local Euler truncation

One Euler step: $x_E = x + \Delta t\,v_1$, $v_1=v(x,t)$.
Heun: $x_H = x + \tfrac{1}{2}\Delta t\,(v_1+v_2)$, $v_2=v(x_E,t+\Delta t)$.
Then

$$
x_H - x_E = \tfrac{1}{2}\Delta t\,(v_2-v_1) = \tfrac{1}{2}(\Delta t)^2\,\hat{a}.
$$

So $\lVert x_H-x_E\rVert$ and $\lVert\hat{a}\rVert$ carry the same geometric information;
controlling one controls the other.

## 4. Equal-error step sizes

Local Euler truncation is $\tfrac{1}{2}(\Delta t)^2\lVert a\rVert$. Holding that error
constant across steps requires

$$
\Delta t \propto \lVert a\rVert^{-1/2}.
$$

The written law $\Delta t=\eta/(\lVert a\rVert+\varepsilon)$ is the $p=1$ special case of
$\Delta t\propto\lVert a\rVert^{-p}$ and does **not** equalize Euler error.
For an order-2 Heun controller on the embedded gap $\delta=\lVert x_H-x_E\rVert$, the
classical PI exponent is $1/(p+1)=1/3$.

**Dimension.** $\lVert a\rVert_2$ scales like $\sqrt{d}$. We therefore use the RMS

$$
\lVert a\rVert_{\mathrm{rms}} = \lVert a\rVert_2/\sqrt{d}
$$

so that $\eta$ can transfer from 2D to images.

## 5. Affine CFG and acceleration capping

Classifier-free guidance is affine in the scale $w$:

$$
v(w)=v_\varnothing + w\,(v_c-v_\varnothing).
$$

The same finite-difference construction is affine, so once both branches are
cached at the predictor and corrector states,

$$
a(w)=a_\varnothing + w\,(a_c-a_\varnothing)
$$

costs no extra network evaluations. The constraint $\lVert a(w)\rVert_{\mathrm{rms}}\le\alpha$
with $w\in[1,w_{\max}]$ is then a one-dimensional clipping problem:

$$
w^\star=\mathrm{clip}\Biggl(
\frac{\alpha\cdot\mathrm{sign}-\langle a_\varnothing,a_\Delta\rangle_{\mathrm{rms}}}
{\lVert a_\Delta\rVert_{\mathrm{rms}}^2+\varepsilon},\;1,\;w_{\max}\Biggr)
$$

when a feasible root exists; otherwise take the endpoint of $[1,w_{\max}]$
with smaller $\lVert a\rVert$. The heuristic $w/(1+\gamma\lVert a\rVert^2)$ is an approximate
soft version of the same idea and creates a fixed-point in $w$ unless $a$ is
lagged from the previous step.

## 6. Fixed-$N$ allocation

A free adaptive controller does not spend a prescribed budget $N$. For fair
ablations we propose a candidate

$$
\Delta t^{\mathrm{prop}}=\frac{\eta}{\lVert a\rVert_{\mathrm{rms}}^p+\varepsilon},
$$

blend it with the remaining uniform budget

$$
\Delta t=\mathrm{clip}\Biggl(
\tfrac{1}{2}\bigl(\Delta t^{\mathrm{prop}}+(1-t)/n_{\mathrm{left}}\bigr),\;
\Delta t_{\min},\;\min(\Delta t_{\max},1-t)\Biggr),
$$

and force the last step onto $t=1$. This keeps exactly $N$ accepted steps
while still concentrating resolution where $\lVert a\rVert$ is large.
