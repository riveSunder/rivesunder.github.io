---
layout: post
title: "Growing Pokemon with Differentiable Cellular Automata" 
date: 2020-12-19 00:00:00 +0000
categories: cellular automata, differentiable programming
---

# Growing Pokemon with Differentiable Cellular Automata

Go back to the 1980s and you'll find that alongside a vibrant wave of connectionism that laid the foundations for the deep learning renassance of the 2010s there was an enthusiastic community of researchers working on cellular automata, which will often refer to as CA in this essay. They were mentioned in Richard Feynman's keynote "Simulating Physics with Computers" ([Feynman 1981](https://www.semanticscholar.org/paper/Simulating-physics-with-computers-Feynman/529595f0bbf7d8d38354436f5ce7a3293e66bd05)) as a candidate computational paradigm for physics models. In particular Feynman liked the locality properties of CA. In general each cell only receives information from a local neighborhood, and this characteristic is considered attractive for scaling purposes. Communication represents one of the largest components of the energy budget for modern computation ([Miller 2017](https://arxiv.org/abs/1609.05510v2)). A practical consequence of long interconnects in computers is that for deep learning models memory access can dominate total energy consumption—squeezing large neural networks to the point that they can fit onto on-chip SRAM can decrease energy costs for memory reads by about 120 times over uncompressed models that require reading in parameters from DRAM ([Han _et al._ 2016](https://arxiv.org/abs/1510.00149)). That's one reason why systolic arrays make attractive building blocks for machine learning accelerators like [Google's line of TPUs](https://cloud.google.com/blog/products/gcp/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu) (Kung 1982 [pdf](https://www.cs.virginia.edu/~smk9u/CS4330S19/kung_-_1982_-_why_systolic_architectures.pdf)). Systolic arrays, like CA systems, are made up of many relatively self-contained compute units each with their own memory and simple processor capabilities. Connections in systolic arrays are local, and just like the abstraction of CA, information flows through many cells, undergoing computation all the while.

There are marked mathematical similarities between CA and neural networks. In fact, work from William Gilpin showed that generalized cellular automata can be readily represented in a special configuration of convolutional neural network ([Gilpin 2019](https://arxiv.org/abs/1809.02942v1)). That's reflected in practice in how modern CA researchers use deep learning libraries like TensorFlow or PyTorch for both computational speedups and automatic differentiation. Despite the shared formulation and convenience of implementing CA with neural network primitives, it's supposedly somehow still difficult for neural networks to learn the most famous set of CA rules of all, John H. Conway's [Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) ([Springer and Kenyon 2020](https://arxiv.org/abs/2009.01398)).

Like other AI-adjacent topics, the roots of cellular automata go back much further in time than contemporary research trends, with John Von Neumann introducing his 29-state rules in 1966 ([Von Neumann and Burks 1966](https://archive.org/details/theoryofselfrepr00vonn_0)) and Conway's famous Game of Life described in Scientific American by Martin Gardner in 1970 (Gardner 1970 [pdf](https://web.stanford.edu/class/sts145/Library/life.pdf)). We'll leave the historical treatment for another time, but it's worth being aware of the rich legacy of CA work and discoveries by hobbyists and professional researchers alike. 

In this essay we are concerned with the synergistic combination of CA and differentiable programming. Unlike classical CA, these systems are parametrized with continuous-valued parameters. Recent work has demonstrated continuous-valued, differentiable CA as a model for development and regeneration loosely based on biological embryogenesis ([Mordvintsev _et al._ 2020](https://distill.pub/2020/growing-ca/)) and "self-classifying" MNIST digits ([Randazzo _et al._ 2020](https://distill.pub/2020/selforg/mnist/))—an approach to image classification that naturally lends itself to semantic segmentation ([Sandler _et al._ 2020\*](https://arxiv.org/abs/2008.04965)). I'll argue here that differentiable CA (also referred to as neural CA, or NCA, in the literature) represent a promising, albeit nascent, research direction that is currently under-appreciated. 

CA have the follwing, non-comprehensive, desirable characteristics that make them attractive as a potential alternative to conventional deep learning models:

* The models tend to be smaller, with comparatively few parameters describing a single set of rules applied repeatedly.

* They are ideally suited for efficient computation on present and upcoming machine learning hardware accelerators, and conveniently suited for implementation with existing deep learning libraries.

* They offer computational flexibility in that computation costs can be balanced against accuracy or uncertainty thresholds by dynamically adjusting the number of CA steps.

In short I think that CA are a reasonable area of research to pursue with strong potential for applied utility and basic AI research. As a "path less traveled" it should at the least yield useful cognitive tools for thinking about learning and intelligence, even if CA can be determined to be less effective or efficient in most use cases. On the other hand, if simpler, smaller CA models can perform some tasks at similar levels to over-parametrized deep learning models, it's a worthwhile pursuit for mitigating issues of environmental impact and societal inequality that can be exacerbated by modern machine learning practice. For examples of what I mean by the aforementioned negative side-effects (or direct effects in some cases) of machine learning, consider that Emma Strubell and colleagues estimated the energy requirements of training a large NLP model with hyperparameter and architecture search has a carbon impact of about 5 times that of an average car _over its entire lifetime_ ([Strubell _et al._ 2019](https://arxiv.org/abs/1906.02243)) and that estimates for monetary costs (at retail price) of compute used for headline deep learning projects [typically](https://www.yuzeh.com/data/agz-cost.html) [fall](https://lambdalabs.com/blog/demystifying-gpt-3/) typically drift into the ten million US dollar range.

## Training Differentiable Cellular Automata

In this essay we'll describe training differentiable CA for constructing images of Pokemon from noisy heavily truncated initial conditions. The training regimen is similar to that described by Mordvintsev _et al._, with a CA implementation in PyTorch is structurally similar to that of Gilpin. CA rules are represented by a set of two convolutional layers with 1x1 kernels, and each cell in the CA state grid is defined by a 16-element vector, with the first 4 elements being interpreted as RGB intensity and transparency. I used alive masking and stochastic cell updates as in Mordvintsev _et al._, details which will be further explained as we get into the details of the experiment. 

<img src="./assets/ca_convolutions.png">

### Training perturbations

During training input images are subjected to 3 types of perturbation: additive noise, truncating the image according to a maximum radius about the image center, and ablating circles of pixels around random coordinates. For the work described here, perturbations are applied only at the first time step, but we could conceivably train more resilient CA by applying perturbations at random intervals.

Departing slightly from the training described in (Mordvintsev _et al._ 2020), I use an incremental difficulty that is modulated in keeping with improved CA performance. This is accomplished by increasing the severity of input image perturbations when the model passes a performance threshold, as well as increasing the number of steps. In the beginning, the input image is essentially a slightly noisy version of the target image, and by the end it amounts to a heavily cropped version with substantial noise and missing pixels. You could consider this as an approximation of training from the last few steps at the start, and gradually moving the starting point backward in terms of the number of steps to reach a solution. In my experience so far this helps prevent the CA model from getting stuck in a local minimum (_e.g._ by zeroing out all the cells) from which it doesn't generate a useful error signal sufficient to escape. 

<img src="./assets/ca_perturbations.png">

### CA Model Formulation

In deep learning we see the consequences of relatively long distance communications in that by compressing a model so that it can fit into on-chip SRAM uses about 120 times less energy than reading weights in from DRAM. 

\* Note that apparently the publication of CA image segmentation by Sandler _et al._ on Arxiv predates the publication of self-classifying MNIST digits by Randazzo _et al._ by a few weeks.
