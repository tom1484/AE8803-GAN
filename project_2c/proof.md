$$
\begin{align*}
z_i = R^T \left( f_i - p \right)
\end{align*}
$$

- $R \leftarrow R\ \mathrm{Exp}\left(\delta \phi\right)$ 
  $$
  \begin{align*}
    h\left(x'\right) &= \left[ R\ \mathrm{Exp}\left(\delta \phi\right) \right]^T
    \left( f_i - p \right)
    \approx \left[ R \left(I + \left[\delta \phi\right]_\times \right) \right]^T
    \left( f_i - p \right) \\
    &= h\left(x\right) + \left[\delta \phi\right]_\times^T R^T \left( f_i - p \right)
    = h\left(x\right) + \left[\delta \phi\right]_\times^T h\left(x\right) \\
    &= z_i + \left[\delta \phi\right]_\times^T z_i \\
    \Rightarrow H_{i, R} \cdot \delta \phi &= \left[\delta \phi\right]_\times^T \hat{z}_i
    = -\left[\delta \phi\right]_\times \hat{z}_i = - \delta \phi \times \hat{z}_i \\
    &= \hat{z}_i \times \delta \phi = \left[\hat{z}_i\right]_\times \cdot \delta \phi \\
    \Rightarrow H_{i, R} &= \left[\hat{z}_i\right]_\times
  \end{align*}
  $$

- $p \leftarrow p + R\ \delta p$
  $$
  \begin{align*}
    h\left(x'\right) &= R^T \left( f_i - p - R\ \delta p \right)
    = R^T \left( f_i - p\right) - R^T R\ \delta p \\
    &= h\left( x \right) - \delta p = z_i - \delta p \\
    \Rightarrow H_{i, p} \cdot \delta p &= -\delta p
    \Rightarrow H_{i, p} = -I_3
  \end{align*}
  $$

- $v \leftarrow v + R\ \delta v$
  $$
  \begin{align*}
    h\left(x'\right) &= R^T \left( f_i - p \right) = z_i \\
    \Rightarrow H_{i, v} \cdot \delta v &= 0_{3 \times 1}
    \Rightarrow H_{i, v} = 0_{3 \times 3}
  \end{align*}
  $$

- $f_i \leftarrow f_i + \delta f_i$
  $$
  \begin{align*}
    h\left(x'\right) &= R^T \left( f_i + \delta f_i - p \right)
    = R^T \left( f_i - p\right) + R^T\ \delta f_i \\
    &= h\left( x \right) + R^T\ \delta f_i = z_i + R^T\ \delta f_i \\
    \Rightarrow H_{i, f_i} \cdot \delta f_i &= R^T\ \delta f_i
    \Rightarrow H_{i, f_i} = R^T
  \end{align*}
  $$

- For $f_j$ where $j \neq i$
  $$
  \begin{align*}
    h\left(x'\right) &= R^T \left( f_i - p \right) = z_i \\
    \Rightarrow H_{i, f_j} \cdot \delta f_j &= 0_{3 \times 1}
    \Rightarrow H_{i, f_j} = 0_{3 \times 3}
  \end{align*}
  $$

Overall
$$
\begin{align*}
H_i = \left[\begin{matrix}
	\left[\hat{z}_i\right]_\times & -I_{3} & 0_{3 \times 3} &
	0_{3 \times 3} & \cdots & R^T & \cdots & 0_{3 \times 3}
\end{matrix}\right]
\end{align*}
$$
