---
date: '2026-08-28T21:48:05+08:00'
draft: false
title: 'Different KL Estimator'
params:
  math: true
---
> This article grew out of my study of John Schulman's [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html). It expands on the original discussion with additional derivations and explanations intended to make the underlying ideas more explicit.

# KL Divergence
For two discrete probability distributions $p$ and $q$ over the same sample space, the KL divergence from $p$ to $q$ is
$$
D_{\mathrm{KL}}(p \| q)
= \sum_z p(z) \log \frac{p(z)}{q(z)}
= \mathbb{E}_{z \sim p}\left[\log \frac{p(z)}{q(z)}\right].
$$
KL divergence measures the difference between the two distributions, weighted by outcomes that are likely under $p$. It is always non-negative, but it is not symmetric:
$$
D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)
$$
in general. Therefore, the order of the two distributions matters.

In **reinforcement learning for language models**, let $\pi_\theta$ be the policy being trained and let $\pi_{\mathrm{ref}}$ be the reference policy. We first fix a prompt $x$.
For a response $y=(y_1,\ldots,y_T)$, its probability under $\pi_\theta$ is
$$
\pi_\theta(y \mid x) =
\prod_{t=1}^{T}
\pi_\theta(y_t \mid x, y_{\lt t}).
$$
The KL divergence between the two response distributions is therefore
$$
\begin{aligned}
D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot \mid x)
\,\|\,
\pi_{\mathrm{ref}}(\cdot \mid x)
\right)
&=
\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}
\left[
\log \frac{\pi_\theta(y \mid x)}
{\pi_{\mathrm{ref}}(y \mid x)}
\right] \\
&=
\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}
\left[
\sum_{t=1}^{T}
\log
\frac{\pi_\theta(y_t \mid x, y_{\lt t})}
{\pi_{\mathrm{ref}}(y_t \mid x, y_{\lt t})}
\right].
\end{aligned}
$$
The space of all possible responses is too large to enumerate. In practice, we sample responses from $\pi_\theta$ and evaluate the same tokens under both $\pi_\theta$ and $\pi_{\mathrm{ref}}$. Their log-probability ratios then provide samples for estimating the sequence-level KL divergence. This is where the different KL estimators become useful.

# $k_1$ Estimator
The $k_1$ estimator comes directly from the expectation form of KL divergence. For a sample $z\sim p$, define
$$
k_1(z)=\log\frac{p(z)}{q(z)}.
$$
This is exactly the expression inside the KL expectation:
$$
D_{\mathrm{KL}}(p \| q)
=\mathbb{E}_{z\sim p}\left[\log\frac{p(z)}{q(z)}\right]
=\mathbb{E}_{z\sim p}[k_1(z)].
$$
Given $N$ independent samples $z_1,\ldots,z_N\sim p$, we estimate the KL divergence by averaging their $k_1$ values:
$$
\widehat{D}_{\mathrm{KL}}^{(k_1)}
=\frac{1}{N}\sum_{i=1}^{N}k_1(z_i).
$$

This estimator is unbiased because
$$
\begin{aligned}
\mathbb{E}\left[\widehat{D}_{\mathrm{KL}}^{(k_1)}\right]
&=\mathbb{E}_{z\sim p}[k_1(z)] \\
&=\mathbb{E}_{z\sim p}\left[\log\frac{p(z)}{q(z)}\right] \\
&=D_{\mathrm{KL}}(p \| q).
\end{aligned}
$$
In other words, its average equals the true KL divergence.

However, $k_1$ can have high variance. A single value is positive when $p(z)\gt q(z)$ and negative when $p(z)\lt q(z)$, even though the final KL divergence is always non-negative. The true KL may therefore be the small result left after many positive and negative samples cancel each other out. When $p$ and $q$ are close, each log-ratio usually changes at first order, while their expected KL is only a second-order quantity. The signal we want is small compared with the variation of individual samples, so many samples may be needed for a stable estimate.

In **reinforcement learning for language models**, we sample a response $y\sim\pi_\theta(\cdot\mid x)$. For each generated token, both policies evaluate the same token using the same prompt and prefix:
$$
k_{1,t}
=\log\pi_\theta(y_t\mid x,y_{\lt t})
-\log\pi_{\mathrm{ref}}(y_t\mid x,y_{\lt t}).
$$
The estimator for the complete response is the sum of these token-level log-probability differences:
$$
\begin{aligned}
k_1(y)
&=\sum_{t=1}^{T}k_{1,t} \\
&=\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x).
\end{aligned}
$$
Averaging $k_1(y)$ over responses sampled from $\pi_\theta$ estimates $D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$. Dividing the sum by $T$ instead gives the length-normalized, per-token value often used in practice.

# $k_2$ Estimator
Let
$$
\Delta(z)=\log\frac{p(z)}{q(z)}.
$$
The $k_2$ estimator is
$$
k_2(z)=\frac{1}{2}\Delta(z)^2
=\frac{1}{2}\left(\log\frac{p(z)}{q(z)}\right)^2.
$$
It comes from a Taylor approximation. Because $p$ and $q$ are probability distributions,
$$
\mathbb{E}_{z\sim p}\left[\frac{q(z)}{p(z)}\right]
=\mathbb{E}_{z\sim p}\left[e^{-\Delta(z)}\right]
=1.
$$
When $p$ and $q$ are close, $\Delta(z)$ is small, so
$$
e^{-\Delta}\approx 1-\Delta+\frac{1}{2}\Delta^2.
$$
Taking the expectation of both sides gives
$$
\mathbb{E}_{z\sim p}[\Delta(z)]
\approx
\mathbb{E}_{z\sim p}\left[\frac{1}{2}\Delta(z)^2\right].
$$
The left-hand side is $D_{\mathrm{KL}}(p\|q)$, which gives $k_2$ as an approximation to KL divergence.

Unlike $k_1$, $k_2$ is biased because the Taylor expansion is only an approximation. Its bias is usually small when $p$ and $q$ are close, but it can become inaccurate when the two distributions are far apart. In return, $k_2$ is always non-negative and usually has lower variance when the distributions are close, so its sample average tends to be more stable.

In **reinforcement learning for language models**, define the token-level log-probability difference
$$
\Delta_t
=\log\pi_\theta(y_t\mid x,y_{\lt t})
-\log\pi_{\mathrm{ref}}(y_t\mid x,y_{\lt t}).
$$
If the complete response is treated as one sample, first sum the token-level differences and then square the result:
$$
\begin{aligned}
\Delta(y)&=\sum_{t=1}^{T}\Delta_t, \\
k_2^{\mathrm{seq}}(y)&=\frac{1}{2}\Delta(y)^2
=\frac{1}{2}\left(\sum_{t=1}^{T}\Delta_t\right)^2.
\end{aligned}
$$
In LLM training code, a token-level form is also common:
$$
k_{2,t}=\frac{1}{2}\Delta_t^2.
$$
These token-level values can be summed or averaged over the response. This is different from squaring the sequence-level sum:
$$
\frac{1}{2}\left(\sum_{t=1}^{T}\Delta_t\right)^2
\neq
\sum_{t=1}^{T}\frac{1}{2}\Delta_t^2.
$$
The sequence-level form treats the complete response as one sample, while the token-level form estimates the KL contribution at each generation step separately and is more common in language-model RL.

# $k_3$ Estimator
We now want an estimator that keeps the unbiasedness of $k_1$ while reducing its variance. A common way to do this is to add a **control variate**: a term whose expectation is zero. Recall that
$$
\Delta(z)=\log\frac{p(z)}{q(z)}.
$$
Because
$$
\mathbb{E}_{z\sim p}\left[e^{-\Delta(z)}\right]
=\mathbb{E}_{z\sim p}\left[\frac{q(z)}{p(z)}\right]
=1,
$$
the term $e^{-\Delta(z)}-1$ has expectation zero. Therefore, for any constant $\lambda$,
$$
\widetilde{k}(z)
=\Delta(z)+\lambda\left(e^{-\Delta(z)}-1\right)
$$
is still an unbiased estimator of $D_{\mathrm{KL}}(p\|q)$.

The clever choice is $\lambda=1$, which gives
$$
k_3(z)
=e^{-\Delta(z)}-1+\Delta(z).
$$
To see why this choice is useful, let $\rho(z)=e^{-\Delta(z)}=q(z)/p(z)$. The inequality $\log\rho\leq\rho-1$ gives
$$
k_3(z)=\rho(z)-1-\log\rho(z)\geq0.
$$
Geometrically, this is the vertical distance between $-\log\rho$ and its tangent at $\rho=1$. Since the added term has expectation zero, $k_3$ remains unbiased:
$$
\begin{aligned}
\mathbb{E}_{z\sim p}[k_3(z)]
&=\mathbb{E}_{z\sim p}[\Delta(z)]
+\mathbb{E}_{z\sim p}[e^{-\Delta(z)}-1] \\
&=D_{\mathrm{KL}}(p\|q).
\end{aligned}
$$
When $p$ and $q$ are close, the added term cancels much of the sample-level fluctuation in $k_1$. This gives $k_3$ the useful combination of being unbiased, non-negative, and usually lower variance.

In **reinforcement learning for language models**, using the token-level log-probability difference $\Delta_t$ defined above gives
$$
k_{3,t}=e^{-\Delta_t}-1+\Delta_t.
$$
These token-level values can be summed or averaged over a response. If the complete response is treated as one sample, use $\Delta(y)=\sum_{t=1}^{T}\Delta_t$ instead:
$$
k_3^{\mathrm{seq}}(y)=e^{-\Delta(y)}-1+\Delta(y).
$$
Averaging either form over samples from $\pi_\theta$ gives the corresponding token-level or sequence-level estimate of $D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$.

# Generalization to $f$-Divergences
The same idea can be used to construct estimators for other $f$-divergences. Using the same ratio $\rho(z)=q(z)/p(z)$ with $z\sim p$, write an $f$-divergence in the form
$$
D_f=\mathbb{E}_{z\sim p}[f(\rho(z))],
$$
where $f$ is convex and $f(1)=0$. Since $\mathbb{E}_{z\sim p}[\rho(z)-1]=0$, subtracting any multiple of $\rho(z)-1$ does not change the expectation. Using the slope of $f$ at $1$ gives
$$
\widetilde{f}(\rho)
=f(\rho)-f'(1)(\rho-1).
$$
This estimator is **unbiased** because the subtracted term has expectation zero. It is also non-negative because a convex function always lies above its tangent:
$$
f(\rho)\geq f(1)+f'(1)(\rho-1)
=f'(1)(\rho-1).
$$
Therefore, $\widetilde{f}(\rho)$ is exactly the vertical distance between $f$ and its tangent at $\rho=1$.

For $D_{\mathrm{KL}}(p\|q)$, choosing $f(\rho)=-\log\rho$ recovers $k_3$. For the reverse direction,
$$
D_{\mathrm{KL}}(q\|p)
=\mathbb{E}_{z\sim p}[\rho(z)\log\rho(z)],
$$
so $f(\rho)=\rho\log\rho$ and $f'(1)=1$. The same construction gives the unbiased, non-negative estimator
$$
\rho\log\rho-(\rho-1).
$$
Thus, $k_3$ is one instance of a general method: subtract the tangent of a convex function at $1$ without changing its expectation.
