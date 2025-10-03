# Comparative Analysis of Topology-Aware Pathfinding: Evaluating Persistent Homology Against Traditional Heuristic Methods

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17257031.svg)](https://doi.org/10.5281/zenodo.17257031)

**Author:** Oksana Osidach  
**Student ID:** 411839  
**Degree:** Bachelor  
**University:** [Your University Name]  
**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  

---

## 📄 Abstract

Pathfinding in complex topological spaces presents unique challenges, particularly when traditional heuristics fail to account for underlying topological constraints. This work investigates the application of **topological data analysis (TDA)**, specifically **persistent homology (PH)** computed from Vietoris–Rips complexes (VR complexes), to enhance classical pathfinding algorithms such as **A\***, **Weighted A\***, and **Greedy Best-First Search**.  

Theoretical chapters provide a foundation in graph-based search methods, simplicial complexes, homology, and the principles of PH, establishing a framework for integrating topological information into heuristic design. Experimentally, synthetic point clouds representing surfaces with nontrivial topology — including spheres, tori, Klein bottles, and crosscaps — are used to construct k-nearest neighbor graphs. Persistent 1-dimensional cycles (H1) extracted from these VR complexes inform heuristic penalties, guiding search algorithms around topological obstacles.  

Experiments explore the effects of point cloud density, start and goal selection, and algorithmic parameters, demonstrating the influence of topological features on path selection, connectivity, and computational efficiency. This work highlights how combining theoretical insights from algebraic topology with practical search methods can lead to more robust navigation strategies on complex surfaces.

---

## 📄 Thesis

The full bachelor thesis PDF is published on Zenodo and can be accessed here:  
[https://doi.org/10.5281/zenodo.17257031](https://doi.org/10.5281/zenodo.17257031)

---

## 🔗 Related Resources

- This repository contains the implementation code and is linked to the thesis as a supplementary resource.  
- You can cite the thesis using the Zenodo DOI:  
```text
O. Osidach (2025). "Comparative Analysis of Topology-Aware Pathfinding: Evaluating Persistent Homology Against Traditional Heuristic Methods." Zenodo. https://doi.org/10.5281/zenodo.17257031
