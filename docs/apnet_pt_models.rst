APNet PT Models
===============

This page documents the trainable `apnet_pt` model families exposed by
`train_models.py`. The classes listed here are the high-level training and
inference harnesses around the lower-level `torch.nn.Module` implementations.

Atomic Models
-------------

These models operate on a single monomer graph and predict atom-resolved
properties.

.. list-table:: Atomic model overview
   :header-rows: 1

   * - Model
     - Main inputs
     - Main outputs
     - Dependencies
   * - `AtomModel`
     - Monomer geometry as a PyTorch Geometric atomic graph
     - Atomic charge, dipole, and quadrupole predictions
     - Standalone atom model; no required submodel
   * - `AtomHirshfeldModel`
     - Monomer geometry with reference multipoles and Hirshfeld targets
     - Atomic multipoles plus Hirshfeld volume-ratio and valence-width targets during evaluation
     - Standalone atom model; no required submodel
   * - `AtomTypeParamModel` (`AtomModels`)
     - Monomer graph with atom typing context
     - Per-atom Hirshfeld-volume-ratio and valence-width parameters
     - Standalone atom-type parameter model
   * - `AtomInducedDipoleModel`
     - Monomer graph and either on-the-fly or precomputed HF/VR descriptors
     - Atomic induced-dipole response together with learned atomic features used by AP3-style models
     - Requires an atom-type HF/VR provider unless `precompute_hfvr=True`
   * - `InducedDipoleModel`
     - Monomer graph, HF/VR descriptors, and optionally pretrained multipole features
     - Induced-dipole response with frozen or reused multipole subnetwork components
     - Can depend on both an atom-type HF/VR model and a pretrained `AtomMPNN`

`AtomModel` is the AP-Net2 atomic baseline. It predicts permanent multipoles and
is the default dependency for APNet2-family pair models. `AtomHirshfeldModel`
extends the monomer target space toward Hirshfeld-derived descriptors.

`AtomTypeParamModel` in `apnet_pt.AtomModels.ap3_atomtype_mpnn` is the AP3
atom-level parameter model. It predicts the Hirshfeld volume ratios and valence
widths later consumed by induced-dipole and fused AP3 workflows.

`AtomInducedDipoleModel` adds polarization-aware monomer learning. It needs
Hirshfeld/valence-width information either from a pretrained atom-type model or
from a dataset that has those quantities precomputed. `InducedDipoleModel` goes
one step further by optionally freezing or reusing a pretrained `AtomMPNN`
submodel so permanent multipoles come from a prior atom model while the induced
response is trained on top.

Pairwise Models
---------------

These models operate on dimers and predict interaction energies or interaction
parameters.

.. list-table:: Pairwise model overview
   :header-rows: 1

   * - Model
     - Main inputs
     - Main outputs
     - Dependencies
   * - `APNet2Model`
     - Dimer graph, with monomer atomic features either precomputed or supplied by an atom model
     - SAPT component energies; common inference returns electrostatics, exchange, induction, and dispersion
     - Usually depends on `AtomModel` / `AtomMPNN`
   * - `APNet2_AM_Model`
     - Dimer graph with fused atom-model features
     - Same APNet2 component-energy targets with atom model folded into the fused architecture
     - Uses the atom model internally as part of the fused forward pass
   * - `dAPNet2Model`
     - Dimer features derived from a pretrained APNet2 pipeline, optionally filtered by level-of-theory tags
     - Delta correction on top of APNet2-style interaction representations
     - Depends on a pretrained `APNet2Model`; often also uses a pretrained atom model
   * - `AM_DimerParam_Model`
     - Dimer geometry plus atom-level parameters or multipoles
     - Classical dimer parameters or energy terms, depending on `dimer_eval_type`
     - Uses an atom-level model and wraps a `DimerProp` evaluator
   * - `AtomTypeParamModel` (`AtomPairwiseModels`)
     - Monomer geometry with a pretrained atom model
     - Atom-type parameters used later by classical dimer-property models
     - Depends on an atom model such as `AtomModel` or `AtomHirshfeldModel`
   * - `APNet3_AtomType_Model`
     - Dimer graph plus atom-type and dimer-property features, optionally with precomputed classical terms
     - AP3-style interaction energies for total-component or FSAPT-style datasets
     - Depends on an atom-type parameter model and a dimer-property model
   * - `APNet3_AtomType_Model` (`fused variants`)
     - Same core inputs as fused APNet3 with different default widths/cutoffs
     - Variant AP3 interaction predictions with the same target families
     - Same dependencies as fused APNet3

`APNet2Model` is the standard learned pairwise SAPT model. In practice it
expects either a pretrained atom model checkpoint or batches where atomic
multipoles and embeddings have already been prepared by the dataset pipeline.

`APNet2_AM_Model` keeps the APNet2 target space but fuses the atomic model into
the pair model. This is the version used when `--train_apnet APNet2-fused` is
selected in `train_models.py`.

`dAPNet2Model` is a delta-learning refinement model. The training script builds
it around an APNet2 harness with hidden states enabled, so the delta model can
learn a correction rather than the full interaction from scratch.

`AM_DimerParam_Model` and the pairwise `AtomTypeParamModel` live in
`apnet_pt.AtomPairwiseModels.mtp_mtp`. They are classical-property models that
bridge learned atom parameters to dimer electrostatics, damping, and induced
dipole style evaluations. These are also the main submodels used to assemble the
fused APNet3 workflows.

The fused APNet3 classes combine a learned pair network with classical
submodels. They require:

- an atom-type parameter model for Hirshfeld-volume-ratio and valence-width-like inputs
- a dimer-property model for classical electrostatic and polarization terms
- either raw or precomputed classical features depending on `use_precomputed_classical`

The `apnet3_fused` class is the main fused AP3 implementation, while
`apnet3_fused_variants` exposes the same harness name with larger default model
sizes and cutoff settings.

Reference Classes
-----------------

- :doc:`autosummary/apnet_pt.AtomModels.ap2_atom_model.AtomModel`
- :doc:`autosummary/apnet_pt.AtomModels.ap2_hirshfeld_atom_model.AtomHirshfeldModel`
- :doc:`autosummary/apnet_pt.AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel`
- :doc:`autosummary/apnet_pt.AtomModels.ap3_atom_model.AtomInducedDipoleModel`
- :doc:`autosummary/apnet_pt.AtomModels.ap3_atom_model_frozen.InducedDipoleModel`
- :doc:`autosummary/apnet_pt.AtomPairwiseModels.apnet2.APNet2Model`
- :doc:`autosummary/apnet_pt.AtomPairwiseModels.apnet2_fused.APNet2_AM_Model`
- :doc:`autosummary/apnet_pt.AtomPairwiseModels.dapnet2.dAPNet2Model`
- :doc:`autosummary/apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model`
- :doc:`autosummary/apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel`
- :doc:`autosummary/apnet_pt.AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model`
- :doc:`autosummary/apnet_pt.AtomPairwiseModels.apnet3_fused_variants.APNet3_AtomType_Model`
