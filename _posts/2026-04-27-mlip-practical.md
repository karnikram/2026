---
layout: distill
title: Evaluating Machine-Learned Inter-Atomic Potentials for a Practical Simulation Workflow
description: MLIPs are a promising new paradigm in atomistic simulation, potentially offering the accuracy of ab-initio methods at the speed of empirical potentials. In this blog post, we give an overview of recent MLIP architectures, followed by an evaluation on a practical CO2 adsorption simulation. We find that as of today these models, though promising, are far from plug-and-play, requiring significant engineering effort to operate within established simulation frameworks, while also failing to produce physically consistent results.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2026-04-27-mlip-practical.bib

toc:
  - name: Introduction
  - name: Overview of MLIPs
    subsections:
    - name: SchNet (2018)
    - name: NequIP (2022)
    - name: Orb-v2 (2024)
    - name: Orb-v3 (2025)
    - name: eSEN (2025)
    - name: UMA (2025)
  - name: MD Frameworks
    subsections:
    - name: ASE
    - name: LAMMPS
      subsections:
      - name: Integrating MLIPs into LAMMPS
  - name: Evaluation
    subsections:
    - name: Inference Speed
    - name: Simulating MOF Adsorption
  - name: Conclusion
  - name: References

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction

Driven by our curiosity about the nanoscopic world,
simulating atoms -- and by extension molecules -- at the quantum level
is a highly active area of research going back a century<d-cite key="lennard1924determination, hohenberg1966inhomogeneous"></d-cite>. From materials science to drug discovery, biophysics, fusion research, and catalysis: accurate tools capable of simulating millions of atoms for nanoseconds within a practical timeframe
could have profoundly positive implications from academia and industry to society at large<d-cite key="fu2023solvent, buehler2004atomic, perilla2017physical, nordlund2006atomistic, yu2023constantpotential, johansson2022micronscale"></d-cite>.
Among the particularly interesting simulation candidates are MOFs, capable among other things
of actively absorbing carbon dioxide for carbon capture to storing hydrogen<d-cite key="sumida2012carbon, chen2020balancing"></d-cite>.
Traditionally, two options exist for molecular dynamics (MD) simulations: Use *prohibitively expensive* ab-initio (from first principles) methods for quantum-level accuracy,
or settle for *significant inaccuracy* by using inexpensive empirical potentials<d-cite key="burke2012perspective, lennard1924determination"></d-cite>.
In recent years MLIPs have emerged, striving to offer the speed of empirical potentials
while maintaining the accuracy of ab-initio methods like density functional theory (DFT),
essentially providing the best of both worlds<d-cite key="goeminne2023dft, SchNet, schutt2021painn"></d-cite>.
However, their practical applicability is still an open question of ongoing research in the field. The general black-box nature of all ML models, with the complex architectures, specialized training procedures and datasets of MLIPs,
paired with their slow integration into commonly used MD software programs pose a significant barrier for researchers and practitioners to use, understand and adopt this promising new paradigm.

In this blog post, we will give an overview of the most important MLIPs and their architectures, followed by a discussion of their practical applicability in terms of their speed, compatibility with commonly used MD software programs and their accuracy on a practical example.

## Overview of MLIPs

Even though DFT methods heavily improve speed and scaling properties for AIMD compared to SOTA coupled cluster methods, they're still too slow for large scale systems, usually the more interesting ones.
One key problem is the fact that DFT approximates the entire wave function, in essence solving the entire system completely,
every single time step, which is very costly and scales poorly with system size.
But for MD we only care about the forces acting on the atoms, which are derived from the wave function, not the wave function itself.
So if there was a way to compute the forces from atomic positions directly, we could skip the highly expensive wave function calculation entirely. MLIPs are trained on a dataset of DFT calculations to learn this mapping as accurately as possible, potentially providing the best of the two worlds: the accuracy of DFT with the speed of cheap empirical potentials.

**Explicit MLIPs** These take as input, for $N$ atoms, their positions $R \in \mathbb{R}^{N \times 3}$ and species $Z \in \mathbb{Z}^{N}$,
and learn the mapping to energy $E \in \mathbb{R}$, forces $F \in \mathbb{R}^{N \times 3}$ and stress $\sigma \in \mathbb{R}^{N \times 6}$ directly.
This is considered the straightforward and intuitive approach, but no energy conservation guarantees are offered by it.

$$
    F_i, E, \sigma_{\alpha \beta} = f_\theta(r_1, r_2, \dots, r_n)
$$

These MLIPs are also referred to as direct-force predictors.

**Implicit MLIPs** These take as input, for $N$ atoms, their positions $R \in \mathbb{R}^{N \times 3}$ and species $Z \in \mathbb{Z}^{N}$,
and a mapping is learned to the potential energy $E \in \mathbb{R}$ of the system alone:

$$
    E = f_\theta(r_1, r_2, \dots, r_n)
$$

The forces $F_i$, the stress $\sigma$, and the strain tensors $\epsilon_{\alpha \beta}$ are then computed by taking
the negative gradient of the network with respect to the atomic positions $r_i$ and the strain tensors $\epsilon_{\alpha \beta}$ respectively.

$$
    \mathbf{F}_i = - \nabla_{\mathbf{r}_i} E
    \quad\text{and}\quad
    \sigma_{\alpha \beta} = \frac{1}{V} \frac{\partial E}{\partial \epsilon_{\alpha \beta}}
$$

They are also called conservative MLIPs. <d-cite key="chmiela2017mlff"></d-cite> were the first to describe this idea in their gradient-domain MLIP approach.
The big problem with this approach is the second backpropagation step, which is very costly, especially for training where you now need a gradient of a gradient.

### SchNet (2018)

SchNet<d-cite key="SchNet"></d-cite> is one of the pioneering GNN-based MLIPs.
Introduced by Schütt et al., it was the first popular implicit MLIP model,
a concept that all major models still more or less employ today.
Its release initiated a paradigm shift in the field, resulting in the gradual replacement of
handcrafted methods like GAP<d-cite key="GAP"></d-cite> by end-to-end fully learnable architectures.

{% include figure.liquid
  path="assets/img/2026-04-27-mlip-practical/schnet_architecture.png"
  class="img-fluid"
  caption="SchNet architecture overview. CFConvs make the surface smooth; the use of distances makes the model invariant to rotations<d-cite key='SchNet'></d-cite>."
%}

#### Invariance to E(3)

SchNet is invariant to rotations, translations, and atomic indexing, which made it a significant step forward compared to previous models.
Translation equivariance and atomic indexing symmetry are the easiest to achieve, as they often come naturally when working with graphs;
which is why, going forward, we will only focus on rotational equivariance.

SchNet achieves rotational equivariance by using the distance $d_{ij} = ||r_i - r_j||_2$ between two atoms $i$ and $j$ as input to the model, an operation *invariant* to rotations.
These distances are then transformed into coefficients of Radial Basis Functions (RBFs),

$$
    e_k(d_{ij}) = \exp(-\gamma (d_{ij} - \mu_k)^2)
    \quad\text{with}\quad
    k \in [1, K],
$$
where $\mu_k$ are the Gaussian centers, a fix for the high variance of the distance variable slightly resembling Fourier Features<d-cite key="tancik2020fourier"></d-cite>,
which are then used further downstream for the update step.

The update step is a simple convolution with ResNet<d-cite key="he2015resnet"></d-cite> skip-connections over neighboring atoms,

$$
    v_i^{(k+1)} = v_i^{(k)} + (v^{(k)} \ast W^{k}) = v_i^{(k)} + \sum_{j \neq i} W^{k}(d_{ij}) v_j^{(k)},
$$

weighted by a learned function $W(d_{ij})$ of the inter-atomic distances.
But newer models no longer take distances alone as model input, due to the fact that they underrepresent atomic systems
especially when the model has fewer interaction layers.
Only measuring distance can lead to having different atomic systems that
result in the same prediction, even though they exert different forces and energies.

#### Continuous Filter Convolutions

A major challenge for applying convolutions in MLIPs is the irregular and sparse arrangement of neighboring atoms.
This contrasts with CNNs for images, where pixels form a regular grid and the discretization
is fine enough to make reasonable predictions.
For MLIPs, a discrete convolution kernel analogous to those in CNNs results in a poorly discretized, discontinuous PES.
SchNet addresses this problem by introducing CFConv layers.
Instead of using a discrete kernel matrix, this method learns a \emph{continuous kernel function}.
First, the distance between two atoms $d_{ij}$ is expanded using a set of RBF as shown above.
Then, the resulting vector is processed by a small MLP to generate a filter weight $W(d_{ij})$:

$$
    W(d_{ij}) = \text{MLP}(e_1(d_{ij}), e_2(d_{ij}), \dots, e_K(d_{ij})).
$$

This technique of generating filter weights from interatomic distances results in a smooth PES,
as illustrated below.

{% include figure.liquid
  path="assets/img/2026-04-27-mlip-practical/schnet_discrete_continuous.png"
  class="img-fluid"
  caption="Discrete convolution kernels result in a low-resolution PES, while continuous convolution filters lead to a smooth one<d-cite key='SchNet'></d-cite>."
%}

### NequIP (2022)

NequIP<d-cite key="batzner2022e3nequip"></d-cite> from Batzner et al. were the first to fully embrace the work
from Tensor Field Networks (2018)<d-cite key="thomas2018tensor"></d-cite> efficiently.
Instead of using just a few equivariant operations, they employ full SO(3) tensor products
using the e3nn<d-cite key="geiger2022e3nn,thomas2018tensor,weiler20183dsteerablecnns,kondor2018clebschgordannets,e3nn_software"></d-cite>
python library.
Nvidia has since released *cuEquivariance*<d-cite key="cuEquivariance"></d-cite>, essentially a GPU-accelerated version of *e3nn*.

The key idea is to project the directional information onto spherical harmonics $$Y_m^{(l)}(\hat{r}_{ij})$$,
which are rotationally equivariant.
These spherical harmonics are then multiplied with a learnable radial function $$R(r_{ij})$$

$$
    S_m^{(l)}(\hat{r}_{ij}) = R(r_{ij}) Y_m^{(l)}(\hat{r}_{ij}),
$$

to form a convolutional filter similar to SchNet's CFConv layers. In their experiments they found that using $l=0$ for the spherical harmonics (i.e. rotational invariance) yielded
significantly worse results than using $l=1$ or $l=2$, but increasing $l$ further didn't improve results much more.

{% include figure.liquid
  path="assets/img/2026-04-27-mlip-practical/nequip_architecture.png"
  class="img-fluid"
  caption="NequIP architecture overview. The use of spherical harmonics allows the model to be fully equivariant<d-cite key='batzner2022e3nequip'></d-cite>."
%}

The Clebsch-Gordan tensor product of two SO(3) representations is again a representation of SO(3), which allows them to mix and match features of different orders $l$ freely - a significant improvement over the use of only a limited set of equivariant operations.

### Orb-v2 (2024)

The Orb models from <d-cite key="neumann2024orbv2"></d-cite> were the first to demonstrate
competitive accuracy without an equivariant or invariant architecture.
They use a denoising diffusion objective similar to DDPMs for pretraining on equilibrium structures,
and then fine-tune on forces and energies as described by <d-cite key="zaidi2022pretraining"></d-cite>.
This makes the model not only more data efficient, but also results in significantly faster inference times,
as it doesn't use a computationally expensive equivariant architecture.
Harcombe et al.<d-cite key="harcombe2025ontheconnection"></d-cite> reported that their diffusion models needed
50% less training data to reach the same accuracy as their non-diffusion models
that also don't respect symmetries in their experiments. Even though symmetry-agnostic diffusion models are more data efficient and
are arguably more accurate for relaxation tasks,
normal NNPs typically yield better force RMSE values.

Orb-v2 is one of the fastest SOTA MLIP models,
primarily due to its highly simple architecture and relatively small amount of parameters.
It avoids loops and is therefore highly vectorizable, allowing for highly efficient batching and parallelization on modern hardware and
uses torch-scatter<d-cite key="paszke2019pytorch"></d-cite> for message aggregation, which supports compilation via TorchScript<d-cite key="devito2023torchscript"></d-cite>.
They provide models pretrained on the MPtrj dataset<d-cite key="deng2023mptrj"></d-cite>.

### eSEN (2025)

The eSEN model from <d-cite key="fu2025esen"></d-cite> builds upon several earlier works from the authors,
including eSCN<d-cite key="passaro2023reducingso3convolutionsso2"></d-cite> and SCN<d-cite key="zitnick2022sphericalchannelsmodelingatomic"></d-cite>.
It is a conservative MLIP that employs a novel SO(2) equivariant architecture,
which avoids e3nn tensor products<d-cite key="geiger2022e3nn"></d-cite>, while still being equivariant to rotations.

{% include figure.liquid
  path="assets/img/2026-04-27-mlip-practical/esen_architecture.png"
  class="img-fluid"
  caption="eSEN architecture overview: Faster SO(2) convolutions emerge naturally when only one degree of freedom remains<d-cite key='fu2025esen'></d-cite>."
%}

The first key idea is to remove all but one degree of freedom in the message passing step.
Given two atoms $i$ and $j$ with their positions $r_i$ and $r_j$, plus their embedding irreps
with corresponding Wigner-D matrix $D_{R_{ij}}$ performs a rotation $R_{ij}$ that aligns the inter-atomic vector $r_{ij} = r_j - r_i$ with the z-axis.
Generally, with one atom fixed at the center of the coordinate system, the other atom has three degrees of freedom to the first atom left, prompting a full SO(3) equivariant architecture.
Zitnick et al.<d-cite key="zitnick2022sphericalchannelsmodelingatomic"></d-cite> rotate the coordinate system such that atom $j$ lies on the positive z-axis, removing two more degrees of freedom.
Then, the eSEN performs the message passing

$$
    m_{ij} = D_{R_{ij}^{-1}} f_{\text{message}}(D_{R_{ij}} h_i, D_{R_{ij}} h_j, |r_{ij}|, \dots)
$$

rotating the result back to the original reference frame.
The message network still sees the spherical channels displayed relative to its frame of reference and
can therefore still make out relative information of its neighbors but only has to learn this with one varying degree of freedom,
the azimuthal angle $\theta$ around the z-axis, which $f_{\text{message}}$ needs to smoothly convolve over.
Spherical harmonics become now become independent of the azimuthal angle $\theta$ and are therefore reduced to circular harmonics,
which only depend on the polar angle $\phi$ as shown below.

{% include figure.liquid
  path="assets/img/2026-04-27-mlip-practical/eSCN_projection.png"
  class="img-fluid"
  caption="Spherical harmonics $Y_m^{(l)}(\theta, \phi)$ reduce to circular harmonics $C_m^{(l)}(\phi)$ when $\theta$ is removed<d-cite key='passaro2023reducingso3convolutionsso2'></d-cite>."
%}

The second key idea is to make use of the fact that a convolution in the spatial domain is
equivalent to a multiplication in the frequency domain<d-cite key="bocher1906introduction"></d-cite>:

$$
    (f * g)(x) = \int f(y) g(x - y) dy
    \quad\Longleftrightarrow\quad
    \mathcal{F}(f * g)(k) = \mathcal{F}(f)(k) \cdot \mathcal{F}(g)(k)
$$

And because just like fourier coefficients the spherical harmonics are such frequency components,
the convolution of coefficients $x_{l,m,c}$ with filters $h_{l,m,c,l',c'}$ can be expressed as

$$
    y_{l,0,c} = \sum_{l'c'} h_{l,0,c,l',c'} x_{l',0,c'}
$$

where $l$ is the order, $m$ the degree, and $c$ the channel index.
But as the coefficients might be complex,

$$
    \left(\begin{array}{c}
    y_{l,m,c} \\
    y_{l,-m,c} \\
    \end{array}\right)
    =
    \sum_{l'c'}
    \left(\begin{array}{cc}
        h_{l,m,c,l',c'} & -h_{l,-m,c,l',c'} \\
        h_{l,-m,c,l',c'} & h_{l,m,c,l',c'}
    \end{array}\right)
    \left(\begin{array}{c}
        x_{l',m,c'} \\
        x_{l',-m,c'}
    \end{array}\right)
$$

is the general formula they use and provide.

Even though in speed it is ahead of the previous e3nn architectures with their notoriously expensive tensor products,
eSEN trails behind the completely non-equivariant Orb models in speed, at least in theory.
Not only due to its conservative architecture, which requires a second backpropagation step to compute forces and stress from the energy,
but primarily because every message passing step requires a custom Wigner-D matrix for rotations into and out of the local frame of reference, a costly operation.

### Orb-v3 (2025)

The Orb-v3 models from <d-cite key="rhodes2025orbv3"></d-cite> improve on the previous Orb-v2 models.
Keeping the same symmetry-agnostic diffusion pretraining approach,
they heavily improve upon the previous version in both speed and accuracy.

{% include figure.liquid
  path="assets/img/2026-04-27-mlip-practical/orb-v3-architecture.png"
  class="img-fluid"
  caption="Orb v3 architecture overview."
%}

They provide both a conservative and direct-force version of the model, just like they did for Orb-v2.
The explicit, direct-force version of Orb-v3 is distilled from the implicit, conservative model<d-cite key="rhodes2025orbv3"></d-cite>.
Novel to v3 is a clever regularization technique that encourages the network to learn equivariance during training,
with minimal computational overhead during inference.
They craft a fixed SO(3) rotation matrix $R$ that is matrix-multiplied with the atomic position matrix before passing them to the model.
They then use $||\nabla_{R} E ||^2_2$ as an additional loss term,
encouraging the model to ignore rotations of the input to the overall energy prediction.

```python
generator = torch.zeros_like(cell, requires_grad=True)
rotation = rotation_from_generator(generator)
positions = (rotation @ positions[:, :, None])[..., 0]

inputs = [positions, displacement, generator]
grads = torch.autograd.grad(
  outputs=[energy],  # (n_graphs,)
  inputs=inputs,  # (n_nodes, 3)
  grad_outputs=[torch.ones_like(energy)],
  # ...
)

rotational_grad = grads[2]
rotational_grad_rms = torch.linalg.norm(
  rotational_grad,
  dim=(1, 2),
).mean()
loss += rotational_grad_rms * rot_grad_weight
```
<div class="caption">
    Orb-v3 equivariance regularization loss.
    The model is encouraged to ignore small rotations of the input to the energy.
</div>

Additionally, they include angular spherical harmonic embeddings as features and
use 8 Bessel bases instead of the Gaussian RBF expansion<d-cite key="SchNet"></d-cite> from Orb v2. Even though gaussian RBF are $C^\infty$ continuous, they are ill-suited for the task.
Bessel functions are oscillatory and better suited for encoding wave-like functions. 
The overall simple architecture avoids any expensive equivariant operations
and allows a streamlined vectorized implementation.
Additionally, Orb-v3 includes an online-trained confidence head similar to AlphaFold<d-cite key="jumper2021highly"></d-cite>,
that predicts the expected error of the model, allowing the user to filter out potentially bad predictions.

### UMA (2025)

The UMA models from <d-cite key="wood2025uma"></d-cite> are the latest addition to the family of MLIPs.
They improve the eSEN family, by training on a vastly larger and more diverse dataset and by
using a clever architecture change to achieve speedups making it almost competitive with the conservative Orb-v3 models.

{% include figure.liquid
  path="assets/img/2026-04-27-mlip-practical/uma_mole_architecture.png"
  class="img-fluid"
  caption="UMA architecture overview: MoLE use fewer active parameters<d-cite key='wood2025uma'></d-cite>."
%}

They introduce MoLE layers, essentially a mixture of experts<d-cite key="jacobs1991adaptive"></d-cite> variant that
can be pre-computed with a global embedding of the atomic environment and global properties like net charge or spin.
This allows the model to adapt its architecture to the specific task at hand,
reducing the amount of active parameters significantly.
While this works quite well for static MD simulations,
any changes in the atomic environment requires re-computing the networks parameters
which are still the largest in any MLIP to date.
Just like eSEN, they use a conservative implicit architecture,
requiring a second backpropagation step to compute
the forces and the stress from the energy derivative.

A short summary of the different MLIP models is shown in the table below,

<div class="table-responsive">
  <table class="table table-sm table-borderless">
    <thead>
      <tr>
        <th>Model</th>
        <th>Implicit</th>
        <th>Equivariance</th>
        <th>A.P.</th>
        <th>N.L.</th>
        <th>Cutoff</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Orb-v2</td>
        <td>&times;</td>
        <td>&times;</td>
        <td>25.2M</td>
        <td>20</td>
        <td>6 Å</td>
      </tr>
      <tr>
        <td>Orb-v3 Direct</td>
        <td>&times;</td>
        <td>&times;</td>
        <td>25.5M</td>
        <td>120</td>
        <td>6 Å</td>
      </tr>
      <tr>
        <td>Orb-v3 Conservative</td>
        <td>&#10003;</td>
        <td>&times;</td>
        <td>25.5M</td>
        <td>--</td>
        <td>6 Å</td>
      </tr>
      <tr>
        <td>Nequix<d-cite key="geiger2022e3nn"></d-cite></td>
        <td>&#10003;</td>
        <td>e3nn</td>
        <td>708k</td>
        <td>--</td>
        <td>6 Å</td>
      </tr>
      <tr>
        <td>MatterSim v1<d-cite key="yang2024mattersim"></d-cite></td>
        <td>&#10003;</td>
        <td>e3nn</td>
        <td>4.5M</td>
        <td>256</td>
        <td>5 Å</td>
      </tr>
      <tr>
        <td>SevenNet MF-ompa<d-cite key="sevennet_mf_ompa"></d-cite></td>
        <td>&#10003;</td>
        <td>e3nn</td>
        <td>25.2M</td>
        <td>--</td>
        <td>6 Å</td>
      </tr>
      <tr>
        <td>eSEN</td>
        <td>&#10003;</td>
        <td>SO(2) conv</td>
        <td>30.2M</td>
        <td>--</td>
        <td>6 Å</td>
      </tr>
      <tr>
        <td>UMA M</td>
        <td>&#10003;</td>
        <td>SO(2) conv</td>
        <td>50.0M</td>
        <td>--</td>
        <td>6 Å</td>
      </tr>
    </tbody>
  </table>
  <div class="caption">
    Performance-relevant model details,
    whether the model is implicit or explicit,
    type of equivariant architecture used,
    the number of active parameters (A.P.),
    the neighbor limit (N.L.) and
    the cutoff radius used during inference.
  </div>
</div>

## MD Frameworks

### ASE
The Atomic Simulation Environment (ASE) is a molecular dynamics (MD) framework tool-set for Python with a strong focus on simplicity and ease-of-use<d-cite key="larsen2017ase"></d-cite>.
Even though most researchers prefer GROMACS<d-cite key="bekker1993gromacs,abraham2015gromacs"></d-cite> or LAMMPS<d-cite key="thompson2022lamps"></d-cite> for maturity, speed and their community,
ASE's simplicity and the fact that it is written in Python makes it the preferred choice
for machine learning (ML) researchers, as most models are also written in Python.
The key strength of ASE is the *calculator* interface, a universal interface that takes as input an `ase.Atoms` object
containing the atomic positions, species, cell vectors and boundary conditions of the system, and outputs the computed potential energy and forces acting on each atom.
Optionally, they can choose to calculate other properties like stress for NPT simulations or charges for electrostatic interactions.
This allows ASE to integrate potentials and dynamics from the vast majority of MD frameworks, simply by
implementing this interface<d-cite key="ase_calculator_docs"></d-cite> for their software.
Additionally, ASE comes with a DFT calculator interface out of the box: the Grid-based Projector-Augmented Wave (GPAW) package<d-cite key="gpaw2024"></d-cite>.
This allows for easy and frictionless switching between different potentials from empirical to DFT,
allowing for easy benchmarking and evaluation of different MLIP models.

### Integrating MLIPs into ASE
The overwhelming majority of MLIP models provide a ready-to-use interface for ASE
in the form of a calculator<d-cite key="ase_calculator_docs"></d-cite> class<d-cite key="rhodes2025orbv3,batzner2022e3nequip,wood2025uma,yang2024mattersim"></d-cite>.
Because ASE is written in Python, and it's abstaining from performing complicated domain decomposition,
the integration of MLIP models is very straightforward, and an example implementation is available in the general addenda.
After linking the *calculator* described in the ASE section,
ASE calls the `calculate` method and fetches the computed properties as required.

### LAMMPS

LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) is a highly scalable and parallelizable MD
simulation framework leveraging empirical and classical potentials<d-cite key="thompson2022lamps"></d-cite>.
It uses domain decomposition to distribute the workload of spatial domains to different workers<d-cite key="plimpton1995fast"></d-cite>.
This allows systems with millions of atoms to be simulated efficiently, even on longer time scales,
making it a frequent choice for researchers in material science.

**Packages.** A key strength of LAMMPS is its matured ecosystem of around 60 different *packages*<d-cite key="lammps_packages_docs"></d-cite>,
that extend the core functionality to meet a wide variety of demands.
These packages can optionally be compiled along with the core software and provide ad-hoc functionality to the user.
They rarely interact with each other and can often be used independently, making for a highly modular software.
Examples include `KSPACE` with a long range electrostatic solver like particle-particle particle-mesh (PPPM)<d-cite key="eastwood1980p3m3dp"></d-cite>,
the `KOKKOS` package for compilation of established styles<d-footnote>property computations</d-footnote> to CUDA kernels<d-cite key="edwards2014kokkos,trott2022kokkos3"></d-cite>,
and the Machine Learning Interatomic Potentials (ML-IAP) package for interfacing MLIPs, as described later in the ML-IAP usage section.

### Integrating MLIPs into LAMMPS
As of writing this blog post, integrating MLIP models into LAMMPS is still a complicated and tedious task.
There are two big roadblocks one will encounter for all integration pathways:

#### Programming language gap.
Even though the overwhelming majority of LAMMPS is written in C++,
the overwhelming majority of ML models are written in Python, and not all of them use the same framework:
Some researchers use PyTorch<d-cite key="paszke2019pytorch"></d-cite> for its unmatched flexibility and ease of use, while
others might prefer to use JAX<d-cite key="jax2018github"></d-cite> from Google for its superior performance.
So either one ports the entire model into C++ using `torch.jit.compile`,
`jax.jit` or by herself;
or she creates a bridge between the two languages using Cython and
Numpy as C++ array interface for Python code.

The first option, Python to C++ compilation, is definitely faster and easier to use, provided it does actually compile.
For example, when trying to port the *Orb model*<d-cite key="rhodes2025orbv3"></d-cite> using TorchScript<d-cite key="paszke2019pytorch"></d-cite>,
it failed to compile almost all the model's submodules due to unsupported python operations like `*kwargs` arguments and `Optional[]` types,
which are non-trivial to remove in a large codebase in a limited time-frame.
Also, this particular model family makes use of an adaptive graph construction scheme, which also seems to clash with TorchScript.

A C++-Python bridge is naturally slower, but framework-agnostic and significantly more versatile, given
one has time and expertise to write the complicated cython interface, which is
often non-trivial and must be done with the utmost care to avoid memory leaks and segmentation faults.

#### Different force philosophies.
LAMMPS is primarily built around the idea of pair styles or pair potentials<d-cite key="lammps_packages_docs"></d-cite>.
Pair potentials don't immediately compute the per-atom force $F_{i}$ at-once, they compute intermediate forces
$$
F_{ij} = f(r_{ij}, \dots)
$$
*between* two atoms $i$ and $j$, usually as a function of their distance $r_{ij}$ first.
Then they sum up all pairwise forces per atom, resulting in the overall force on an atom $i$<d-cite key="lammps_pair_to_overall_force_rule"></d-cite>:
$$
F_{i} = \sum_{j} F_{ij}
$$

$$
F_{j} = \sum_{i} - F_{ij}
$$

While this approach is mostly reasonable for most empirical potentials,
MLIPs are designed to output the overall per-atom forces $F_{i}$ directly or as a gradient of the potential energy $U$ with respect to the atomic positions $r_{i}$:
And that is not just a different design philosophy, it is the very reason why MLIPs are as accurate as they are.
They should ideally operate holistically in the atoms' neighborhood,
$$
F_{i} = f(\{r_{ij}, r_{ik}, \dots\}, \dots)
$$
instead of focussing just on pairwise interactions that fail to capture important many-body effects like angular dependencies
(e.g. in H<sub>2</sub>O).
Here, $j, k \in N(i)$ denotes all neighbors of atom $i$.
To bridge this gap, two options make themselves available:
In the philosophy of GDML from Chmiela et al.<d-cite key="chmiela2017mlff"></d-cite>, one can task implicit (conservative) models to output pairwise forces<d-cite key="nequip_lammps_autograd_integration"></d-cite>
as a gradient of the potential energy $U$ with respect to the inter-atomic distance vector $r_{ij}$:
$$
F_{ij} = -\frac{\partial U}{\partial r_{ij}} \cdot \frac{r_{ij}}{||r_{ij}||}
$$
which allows one to apply them in the pairwise force philosophy of LAMMPS.
Only a handful of models offer support for this type of readout, as it adds computational overhead and integrational complexity,
and they only do this to support LAMMPS<d-cite key="nequip_lammps_autograd_integration"></d-cite>.
Not only is conformity to pair forces tedious and must be redone for every model, it can result in *significant* performance degradation.
Packages in LAMMPS generally expect pairwise potentials to be fairly inexpensive,
and will not shy away from calling pair potentials as often as they like, in the extreme case once per atom per step<d-cite key="lammps_gcmc_spam_calling_pair_potential"></d-cite>,
which will -- provided the model lacks a smart caching strategy -- result in a slowdown of the already quite expensive grand canonical Monte Carlo (GCMC) step
in the worst-case by a factor of $N$, where $N$ is the number of atoms in the system.
The other possibility is to modify LAMMPS's core to allow for direct per-atom forces,
which is not trivial and requires at least some knowledge of `C++` and LAMMPS's internal workings.

#### Option 1: New LAMMPS package
The most complicated, time-consuming and redundant way of integrating MLIPs into LAMMPS is the creation of a new package
that provides a custom `pair_style`.
This would require writing a new `pair_style` class in C++ that
interfaces with our MLIP models,
either via C++-Python bridge or by porting the model to C++ as discussed earlier in the programming language gap section.

While this is by far the most flexible option with the highest level of control,
it requires the most interation with LAMMPS's core codebase,
is the most time-consuming and might need additional maintenance in the future.
Popular models like MACE<d-cite key="batatia2022mace"></d-cite> were directly ported into C++ and given a custom package<d-cite key="integrating_mace_into_lammps"></d-cite>.
Even though it can be the faster option in terms of MD throughput on the CPU, porting a model requires a lot of work
from experts in both C++ and the respective MLIP framework; a process which is not always possible,
not guaranteed to offer the best performance and rarely graphics processing unit (GPU) compatible out-of-the-box.
All-in-all, this option didn't seem general, realistic or time-efficient enough for our purposes.

#### Option 2: The ML-IAP package
To provide a universal interface for MLIP models, the ML-IAP package was added to LAMMPS in 2020.
It uses a cython C++ to Python bridge to interface with Python code,
allowing for on-the-fly integration of models without the need for a port.
To integrate a brand-new MLIP model into LAMMPS when acting through python,
one can either specify a loaded python module inheriting from the abstract `MLIAPUnifiedInterface` class performing the model invocation,
or specify a `.pt` file containing such conforming model serialized by `torch.save`.
This means that at the time of writing, one can not load a JAX model from a file, although support for it is underway<d-cite key="githublampsjaxpulldraft"></d-cite>.
Again, this operation does not compile the model into C++;
instead it just serializes the model weights and architecture into a Python pickle file.
A schematic overview of the standard, indented ML-IAP integration is shown below.
{% include figure.liquid
  path="assets/img/2026-04-27-mlip-practical/lammps-mliap-integration.png"
  class="img-fluid"
  alt="Block diagram showing the ML-IAP integration pipeline"
  caption="Integrating MLIPs via ML-IAP into LAMMPS. Blue means written in Python, yellow means written in C++."
%}
A big shortcoming of the provided cython bridge is the absence of absolute atomic positions.
One is just given the pairwise distance vectors, which are often not enough.
Not only do almost all models expect to receive absolute atomic positions,
some model families like Orb<d-cite key="rhodes2025orbv3,neumann2024orbv2"></d-cite> actually require them for efficient graph construction.
The `ML-IAP` package typically expects the model to output the pairwise forces $F_{ij}$ instead of the overall per-atom forces $F_{i}$,
as discussed in the different force philosophies section above.
This means that one either folds and rewrites the model to conformity like NequIP<d-cite key="batzner2022e3nequip,tan2025high"></d-cite> did<d-cite key="nequip_lammps_autograd_integration"></d-cite>,
or the package is modified and the complicated compilation process that comes with that.

#### Option 3: Metatomic
In search for integration alternatives, the recent `Metatomic`<d-cite key="metatomic"></d-cite> project by Filippo Bigi et al. <d-cite key="bigi2025metatomic"></d-cite> was a promising option. It
aims to provide a universal MLIP model interface for not only LAMMPS,
but also ASE, GROMACS<d-cite key="bekker1993gromacs,abraham2015gromacs"></d-cite>, the path-integral molecular dynamics driver i-PI<d-cite key="ipi3"></d-cite> and many other frameworks,
meaning that once a model is compatible with `Metatomic`, it can be used seamlessly in all supported frameworks without any additional work.

#### Discussing the Options
Option 3 would have been the most elegant, future-proof and helpful solution.
so it was naturally considered first.
The primary downside is a C++ porting requirement with `torch.jit.compile`,
neither always possible for torch models, nor applicable to `JAX` models like Nequix<d-cite key="koker2025nequix"></d-cite>.
Option 1 was clearly a last resort in case the `ML-IAP` package would turn out unpatchable, which left
us only with option 2, the `ML-IAP` package.

#### Modifying LAMMPS
`ML-IAP` already had access to the `atoms->x` array naturally,
but didn't provide it to the cython interface.
To give the python code access, a pointer to the position array `atom->x` was added
to the `MLIAPData` class, which is then wrapped by a cython interface class `MLIAPDataPy`
which one can then use in python.
For an efficient and zero-copy transfer of the C++ array to python,
numpy<d-cite key="harris2020numpy"></d-cite> is used as an interface.
The only problem was that C++ arrays don't carry information about their length, as they are just pointers to the first element in memory.
So we looked for a variable in LAMMPS that would provide us with the number of atoms in the system, which sounds easy but
was anything but.
`atoms->ntotal` exists, but this number varies over time, even with the same number of atoms as input,
due to -- which took some figuring out -- the presence of *ghost atoms*.
LAMMPS uses so-called ghost atoms for inter-worker communication,
allowing the simulation of larger systems via domain decomposition as described by Plimpton<d-cite key="plimpton1995fast"></d-cite>.

<div class="row mt-3">
  <div class="col-md-6">
    {% include figure.liquid
      path="assets/img/2026-04-27-mlip-practical/lammps-ghost-atoms.png"
      class="img-fluid"
      alt="Ghost atoms in LAMMPS enable periodic boundary conditions"
      caption="Ghost atoms in LAMMPS are used to simulate periodic boundary conditions."
    %}
  </div>
  <div class="col-md-6">
    {% include figure.liquid
      path="assets/img/2026-04-27-mlip-practical/lammps-atoms-memory-layout.png"
      class="img-fluid"
      alt="Memory layout for atom->x and atom->f arrays"
      caption="Memory layout of `atom->x` and `atom->f`, with local atoms first and ghost atoms appended."
    %}
  </div>
</div>
so it wasn't exactly clear what type of atom can be found at what position in the `atoms->x` array.
After some digging through the LAMMPS source code and some trial-and-error,
two facts became clear:
Both `atom->x` and `atom->f` contain the positions and forces of *all atoms*, local and ghost.
Secondly, the arrays start with local atoms, directly followed by all ghost atoms as illustrated in the memory-layout panel above.
We use all atoms, local and ghost as input to our MLIPs, because ghost atoms too can exert forces on real atoms
as shown in the ghost-atom schematic above.
We only apply the predicted forces to our local atoms, and postulate that
if there exists a ghost atom, it is updated by *exactly the same* force predictor.
This is important to ensure that Newton's third law holds, i.e. that every action has an equal and opposite reaction.
As was learned only later, LAMMPS uses ghost atoms not only for inter-worker communication,
but also to simulate periodic boundary conditions, just like in the earlier illustration.
Models are informed that the system is non-periodic and the rest is handled by LAMMPS automatically.
This also means that models that previously didn't support periodic boundary conditions, now do.
Additionally, if one is not interested in the stress tensor, the original cell can simply be replaced by a large cubic cell,
which we did for simplicity and to avoid any potential issues with out-of-cell atoms.
We recognize that this solution requires the MLIP to compute more interactions than strictly necessary,
resulting in an overhead of approximately 20-30%.
Future work could focus on adding support for periodic boundary conditions naturally, ignoring ghost atoms.
For this purpose, cell and boundary information has also been made available in the `MLIAPData` class.

#### All roads lead to ASE

{% include figure.liquid
  path="assets/img/2026-04-27-mlip-practical/lammps-universal-ase-adapter.png"
  class="img-fluid"
  alt="Universal ASE adapter for ML-IAP integration"
  caption="Building a universal ASE adapter for LAMMPS ML-IAP integration. Blue means written in Python, yellow means written in C++."
%}
Pretty much all state-of-the-art (SOTA) models provide an ASE calculator interface, which has become the de-facto standard.
There is no reason to waste time and effort writing complicated integrations for each and every model,
when one can instead just write a small `UniversalASEUnifiedMLIAPInterface` class that combines both worlds
seamlessly.
Note that only the `real` and `metal` LAMMPS units are supported.
ASE uses `metal` units, eV for energies and eV/Å for forces,
while LAMMPS `real` units use kcal/mol for energies and kcal/(mol·Å) for forces,
which we automatically convert using the conversion factor 1 eV = 23.0621 kcal/mol<d-cite key="ev_to_kcalmol"></d-cite>.

## Evaluation

### Inference Speed

We measure inference speeds of force prediction as primary MD task
on homogenous uniformly sampled systems of up 100,000 Helium atoms with an
average density $\rho$ of 0.033 $\text{atoms/Å}^3$. To reduce variance of measurements caused by startup tasks like JIT compilation,
a 40-step warm-up is performed on a system of 5 atoms.
Then, we measure the inference speed of MLIPs on systems of 10, 100, 1.000, 10.000, and 100.000 atoms.
We repeat the measurements a total of 20 times for each system size and model, and discard
the first measurement to reduce size-depended startup tasks found in models like Orb-v3<d-cite key="rhodes2025orbv3"></d-cite>.
To take caching out of the equation, no atomic systems were reused.
While <d-cite key="rhodes2025orbv3"></d-cite> report up to 40\% performance gains from reducing neighbor limits of Orb-v3 to 20,
we assume that any practical application will always prefer to avoid neighbor limits to reduce the risk of energy drift and instabilities in long-running MD simulations.
We use an AMD Ryzen Threadripper PRO 5975WX 32-Core processor
with 256GB of RAM for CPU inference and a NVIDIA® H100 NVL® GPU with 100GB of VRAM
for GPU inference.
When possible, all models were compiled using `torch.compile` through ASE calculator
options and for JAX models using JAX's `jit` compilation.

All models exhibit *linear scaling* with respect to the system size,
meaning that MLIPs have the same computational complexity class as empirical force fields.
Simulating a system twice as large simply requires twice the resources or double the time,
which is not the case for any ab-initio method and a major advantage of MLIPs.
However, the fact that all models scale linearly on the CPU and GPU does not mean
that they are equally fast in practice.
Using a GPU for inference results in speedups two orders of magnitude, as long as the system is larger than about 1.000 atoms.
Inference times for smaller systems are bottlenecked by the general overhead and data transfer to and from the GPU.
So for applications like high-throughput screening batching smaller systems
to a single, clustered system can improve throughput by a large margin.
Even though a single MD run can't be batched due to its sequential nature,
independent parallel-running MD simulations of smaller systems can be batched together and benefit.
But not only the GPU has a large effect on the linear factor:
Orb-v3 Direct predicts forces of 100.000 atoms
as fast as UMA can predict the forces of only about 2.000 atoms. So even though in theory all models scale linearly with system size, O(N); the actual inference speed is model dependent.Explicit models without an equivariant architecture like Orb-v2 and direct Orb-v3 run the fastest.
Even with more parameters than Orb-v2 and a higher neighbor limit, Orb-v3 ends up being slightly faster due to reduced overhead of
edge aggregation and message passing of 10 layers fewer<d-cite key="rhodes2025orbv3"></d-cite>.
On third place is Nequix, that is both implicit and uses the tensor products of e3nn<d-cite key="geiger2022e3nn"></d-cite> for equivariance,
but has a relatively small number of parameters and uses JAX instead of PyTorch.

In the following table we converted the raw inference speed into a more intuitive unit of nanoseconds per day (ns/day),
assuming that a simulation is run with a time step of 1 fs ($10^{-15}$ s).
This is a more common unit to express simulation speeds in the MD community,
as it intuitively relates to how much simulation time can be covered in a day of wall-clock time.

<div class="table-responsive">
  <table class="table table-sm table-borderless">
    <thead>
      <tr>
        <th>Model</th>
        <th>10 atoms</th>
        <th>100 atoms (&darr;)</th>
        <th>1,000 atoms</th>
        <th>10,000 atoms</th>
        <th>100,000 atoms</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Orb-v3 Direct</td>
        <td><strong>16.14</strong></td>
        <td><strong>15.55</strong></td>
        <td><strong>8.95</strong></td>
        <td><strong>1.29</strong></td>
        <td><strong>0.08</strong></td>
      </tr>
      <tr>
        <td>Orb-v2</td>
        <td>11.57</td>
        <td>11.04</td>
        <td>6.79</td>
        <td>0.92</td>
        <td>0.06</td>
      </tr>
      <tr>
        <td>Nequix</td>
        <td>12.00</td>
        <td>8.91</td>
        <td>2.41</td>
        <td>0.25</td>
        <td>--</td>
      </tr>
      <tr>
        <td>Orb-v3 Conservative</td>
        <td>7.46</td>
        <td>7.00</td>
        <td>3.15</td>
        <td>0.36</td>
        <td>--</td>
      </tr>
      <tr>
        <td>MatterSim v1</td>
        <td>5.40</td>
        <td>5.10</td>
        <td>2.84</td>
        <td>0.41</td>
        <td>--</td>
      </tr>
      <tr>
        <td>SevenNet MF-ompa</td>
        <td>1.82</td>
        <td>1.51</td>
        <td>0.51</td>
        <td>--</td>
        <td>--</td>
      </tr>
      <tr>
        <td>eSEN OAM</td>
        <td>1.40</td>
        <td>1.19</td>
        <td>0.24</td>
        <td>--</td>
        <td>--</td>
      </tr>
      <tr>
        <td>UMA M</td>
        <td>0.78</td>
        <td>0.84</td>
        <td>0.16</td>
        <td>0.01</td>
        <td>--</td>
      </tr>
    </tbody>
  </table>
  <div class="caption">
    Inference speeds in ns/day for different models and system sizes on a single H100 GPU,
    sorted by inference speed for 100 atoms. Almost all models ran out of memory for 100,000 atoms.
  </div>
</div>

### Simulating MOF Adsorption

Metal-Organic Frameworks (MOFs) have become highly popular among materials scientists,
primarily due to their exceptional porosity and tunable properties<d-cite key="furukawa2013chemistry"></d-cite>.
They consist of metal nodes (ions or clusters) connected by organic linkers,
resulting in a highly porous structure with an enormous internal surface area.
For example, a single gram of MOF-5 has a surface area of 2900 m$^2$<d-cite key="furukawa2013chemistry"></d-cite>,
which is roughly equivalent to the area of half a soccer field.
These features, among others, make MOFs particular interesting
for gas storage and separation applications.
Specifically, Mg-MOF-74<d-footnote>Aka. CPO-27-Mg or Mg$_2$(DOBDC), DOBDC = 2,5-dihydroxyterephthalic acid<d-cite key="xiao2019mof74"></d-cite></d-footnote>
is of particular interest to the scientific community for its excellent carbon-dioxide adsorption<d-cite key="degaga2022quantum"></d-cite>.

CO$_2$ is the primary greenhouse gas responsible for climate change<d-cite key="ipcc2021climate, houghton2004global, mann1998global, lovelock2006"></d-cite>,
so materials that can efficiently capture and store carbon-dioxide<d-cite key="ipcc2022ccs, sumida2012carbon, li2011carbon"></d-cite> could play a critical role reversing pollution and its effects.
MOFs can also be used to store other gasses like methane and hydrogen<d-cite key="chen2020balancing"></d-cite>,
making them versatile materials for various gas storage applications.
Having fast and accurate potentials to simulate CO$_2$ adsorption in different MOF structures
could greatly accelerate the discovery of new materials for carbon capture applications via HTS.
The goal is to benchmark different MLIP models on their ability to simulate carbon-dioxide adsorption in Mg-MOF-74,
compared to empirical potentials.
The two primary forces acting on the CO$_2$ molecules in the MOF
are VdW forces and electrostatic forces<d-cite key="sumida2012carbon"></d-cite>.

**Electrostatic interactions** are fundamental forces between charged particles and play a critical role in a wide range of physical,
chemical, and biological processes.
These forces can be modelled with Coulomb's law<d-cite key="griffiths1999introduction"></d-cite>, which describes the potential interaction energy between two point charges in a vacuum as

$$
    E_{\text{elec}} = \frac{q_{1} q_{2}}{4 \pi \epsilon_{0} r}
$$

where $q_{1}$ and $q_{2}$ are the charges of the interacting particles, $r$ is their separation distance, and $\epsilon_{0}$ is the permittivity of free space.
The $1/r$ dependence of electrostatic interactions makes them exceptionally long-ranged, meaning they cannot be neglected even at distances exceeding 25 Å.
In our MOF example, the interaction energy between the $Mg^{2+}$ metal node ($q\approx1.45e$<d-cite key="fu2023solvent"></d-cite>)
in Mg-MOF-74 and the carbon atom of a CO$_2$ molecule ($q\approx0.7e$<d-cite key="fu2023solvent"></d-cite>) at a distance of 15 Å remains 0.9710 eV (141 kJ/mol).
The corresponding Coulomb force, which decays with a factor of $1/r^2$, is still 1.07 eV/Å (103 kJ/mol), a non-negligible value.
Nevertheless, MLIPs often restrict their ``receptive field'' to only a few Ångströms primarily for performance
-- typically 6–10 Å<d-cite key="rhodes2025orbv3, neumann2024orbv2"></d-cite>.
Capturing such long-range interactions would solely rely on message-passing propagation,
a process that would require the atoms to be connected through a chain of neighbors.
This can be particularly challenging in porous materials like MOFs, where the MOF atoms are dense (meaning neighborhood limits are quickly reached)
and often far apart from the gas molecules (cutoff distances are easily exceeded).

**Van-der-Waals interactions** are the other primary force acting on CO$_2$ molecules in Mg-MOF-74,
and as discussed in the Kohn-Sham section they require dispersion corrections to be modelled accurately in DFT simulations.
The effects of VdW forces fall off with $1/r^{6}$ and are therefore rather short-ranged compared to electrostatic interactions.
While some models like Orb-v3<d-cite key="rhodes2025orbv3"></d-cite> include dispersion corrections directly in their inference pipeline,
it is possible to add dispersion corrections as a post-processing step to any MLIP model.
In ASE this can be quickly done by jointly adding both model calculator and a dispersion calculator (like d3 for example)
to a `SumCalculator` object that simply sums the energies and forces of both calculators.
Contrasting the need for a post-processing correction, the UMA model family<d-cite key="wood2025uma"></d-cite> is directly trained
on various dispersion-corrected datasets, including `OMol25`<d-cite key="levine2025omol25, uma_huggingface_demo"></d-cite>. This means that UMA has learned to capture dispersion interactions directly from data,
without the need for an explicit dispersion correction.

{% include figure.liquid
  path="assets/img/2026-04-27-mlip-practical/mof-74-mg-structure.jpg"
  class="img-fluid"
  caption="Structure of Mg-MOF-74: Red, cyan, grey, and white spheres represent oxygen, magnesium, carbon, and hydrogen atoms<d-cite key='degaga2022quantum'></d-cite>."
%}

To benchmark MLIPs in a realistic setup, we simulate the adsorption of CO<sub>2</sub> in Mg-MOF-74 via MD simulations.
We closely follow and make use of the scripts provided by Fu et al.<d-cite key="fu2023solvent"></d-cite> and adapt them to work with our MLIP models in LAMMPS,
which previously simulated CO<sub>2</sub> adsorption in Mg-MOF-74 with classical force fields.
Because they make use of LAMMPS for their GCMC simulation, the need for a MLIP bridge emerged which was discussed earlier.
A GCMC simulation is basically just a normal MD simulation with the structure fixed,
with one additional step: every few MD steps, a molecule -- in our case CO<sub>2</sub> -- is either inserted or deleted from the system
based on the chemical potential and temperature.
This allows the system to equilibrate with a gas reservoir at a given pressure and temperature,
simulating the adsorption process accurately.
We simulate 10 different pressure settings from 0 to 1 bar at 300K each for 3,000,000 MD steps with a timestep of 2 fs.
To improve the pressure resolution at lower pressures,
the simulations use logarithmic pressure settings as shown in the table below.
<div class="table-responsive">
  <table class="table table-sm table-borderless">
    <thead>
      <tr>
        <th>Simulation</th>
        <th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Pressure (bar)</td>
        <td>0.000</td><td>0.073</td><td>0.151</td><td>0.237</td><td>0.330</td><td>0.433</td><td>0.547</td><td>0.677</td><td>0.825</td><td>1.000</td>
      </tr>
      <tr>
        <td>Pressure (kPa)</td>
        <td>0.000</td><td>7.3</td><td>15.1</td><td>23.7</td><td>33.0</td><td>43.3</td><td>54.7</td><td>67.7</td><td>82.5</td><td>100.0</td>
      </tr>
    </tbody>
  </table>
  <div class="caption">Pressure sampling points computed as $$1 - \ln(\text{linspace}(e, 1, 10))$$.</div>
</div>

For simplicity, we convert 1 bar to 100 kPa instead of 101.325 kPa during post-processing.
Every 10 simulation steps, we dump the entire system state to a lammpstr file, about 8 GB per simulation.
To compress the output data, we convert the lammpstr files to a custom delta-compressed binary format with
gzip compression, resulting in a trajectory file of about 100MB per simulation.
To compute the equilibrium adsorption capacity, we only consider the last 50\% of each simulation,
and average the number of adsorbed CO<sub>2</sub> molecules over time.

Unfortunately, the adsorption isotherm calculation could only be performed
using the Orb-v3 direct-force model and the original empirical potentials from Fu et al.<d-cite key="fu2023solvent"></d-cite>,
as all other models were simply too slow.

Running the fast direct-force Orb-v3 model on the LAMMPS-extended Mg-MOF-74
carbon-dioxide filled system<d-footnote>with approximately 800 atoms, ghost and local</d-footnote> took about two days on two H100 NVL® GPUs running in parallel
for all 10 pressure points,with a speed of about 24ns/day<d-footnote>8 hours for the 6ns-long simulation</d-footnote>
with a time step of 2 fs.
Using table the inference speed table from the inference speed results,
we estimate other models could take weeks or even months to complete,
even when using multiple H100 NVL® GPUs in parallel.
The results of the two simulations are as follows.

<div class="row mt-3">
  <div class="col-md-6">
    {% include figure.liquid
      path="assets/img/2026-04-27-mlip-practical/co2_density_hexbin_pressure_9_empirical.png"
      class="img-fluid"
      caption="Recreated empirical results<d-cite key='fu2023solvent'></d-cite>"
    %}
  </div>
  <div class="col-md-6">
    {% include figure.liquid
      path="assets/img/2026-04-27-mlip-practical/co2_density_hexbin_pressure_9_direct-orb-v3-omol.png"
      class="img-fluid"
      caption="Orb-v3 direct-force results<d-cite key='rhodes2025orbv3'></d-cite>"
    %}
  </div>
</div>
<div class="caption">
    CO<sub>2</sub> atom density plots in Mg-MOF-74 at 298K at 1 bar pressure.
    Results are obtained from a 1:100 subsample of the GCMC simulation run.
</div>


<div class="row mt-3">
  <div class="col-md-6">
    {% include figure.liquid
      path="assets/img/2026-04-27-mlip-practical/co2_vs_pressure_empirical.png"
      class="img-fluid"
      caption="Recreated empirical results<d-cite key='fu2023solvent'></d-cite>"
    %}
  </div>
  <div class="col-md-6">
    {% include figure.liquid
      path="assets/img/2026-04-27-mlip-practical/co2_vs_pressure_direct-orb-v3.png"
      class="img-fluid"
      caption="Orb-v3 direct-force model results<d-cite key='rhodes2025orbv3'></d-cite>"
    %}
  </div>
</div>
<div class="caption">
    CO<sub>2</sub> adsorption isotherms in Mg-MOF-74 at 298K.
</div>

<div class="row mt-3">
  <div class="col-md-6">
    {% include figure.liquid
      path="assets/img/2026-04-27-mlip-practical/co2_vs_time_all_colored_empirical.png"
      class="img-fluid"
      caption="Recreated empirical results<d-cite key='fu2023solvent'></d-cite>"
    %}
  </div>
  <div class="col-md-6">
    {% include figure.liquid
      path="assets/img/2026-04-27-mlip-practical/co2_vs_time_all_colored_direct-orb-v3.png"
      class="img-fluid"
      caption="Orb-v3 direct-force model results<d-cite key='rhodes2025orbv3'></d-cite>"
    %}
  </div>
</div>
<div class="caption">
    Number of adsorbed CO<sub>2</sub> molecules over time for different pressure settings of the GCMC simulation in Mg-MOF-74 at 298K.
</div>

While the Orb-v3 direct-force omol model can reproduce some of the results from <d-cite key="fu2023solvent"></d-cite>,
it fails to capture the essence of CO$_2$ adsorption in Mg-MOF-74 correctly.
We observed the density plots of the CO$_2$ atoms inside the MOF structure,
where we can see the wave-like patterns originating from the magnesium ions at the corners of the MOF
predicted by the empirical model, with a slight resemblance of this pattern in the Orb-v3 direct-force results.
Additionally, the GCMC algorithm sometimes inserts CO$_2$ molecules directly into the MOF structure itself
for some unknown reason, which might have a negative effect on the results generally.
In the figures above we show the adsorption isotherms predicted by the empirical model (left)
and the Orb-v3 direct-force omol model (right), how many CO$_2$ molecules
stay in the GCMC simulation at different pressures.
While the empirical model predicts a logarithmic increase in adsorbed CO$_2$ molecules with increasing pressure,
the Orb-v3 direct-force omol model predicts a constant number of adsorbed CO$_2$ molecules for all pressure settings.
We fit the Langmuir adsorption model<d-cite key="langmuir1917adsorption"></d-cite>

$$
    q = \frac{q_{\text{max}} b P}{1 + b P}
$$

to both isotherms, where $q$ is the amount adsorbed at pressure $P$, $q_{\text{max}}$ is the maximum adsorption capacity,
and $b$ is a constant related to the affinity of the adsorbent for the adsorbate.
This model is plotted as a dashed line in the adsorption isotherm figure,
showing a good fit for the empirical results
while the Orb-v3 direct-force omol model predicts almost no pressure-related
correlation for CO$_2$ adsorption.
Finally, we plot the number of CO$_2$ molecules inside the GCMC simulation over time
for different pressure settings in the figure above.
While the empirical model shows a small variation in the number of CO$_2$ molecules
over unique pressure-dependent mean values over time,
the results obtained from the Orb-v3 direct-force omol model seem random with a high variance
without any pressure-dependent mean value.
A potential area of future work could be the evaluation
-- with the necessary computational resources at hand --
of more models or orb-v3 variations to find better performing models for this task.


## Conclusion

In this post, we explored how MLIPs work and how they are built,
different architectures and design choices that go into making them.
We were successful in building a universal ASE-MLIP interface for LAMMPS,
allowing us to run an originally empirical Mg-MOF-74 CO$_2$-adsorption GCMC simulation with a MLIP instead, only to discover that it doesn't reproduce the more physically-sound empirical results. While we were able to evaluate the cheapest MLIP model on the CO$_2$ GCMC simulation,
the more accurate models were simply too expensive to run in a reasonable time frame.
