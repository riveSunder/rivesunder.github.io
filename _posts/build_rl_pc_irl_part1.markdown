---

layout: post
title:  "Why Don't You Build a RL Computer?"
date:   2020-04-04 00:00:00 +0000
categories: RL  

---

<blockquote>
<strong>
Deep reinforecement learning and evolutionary algorithms have significantly different computational requirements than deep learning for computer vision. Building your own workstation for RL/EA may be a daunting prospect, but compared against high cpu cloud compute the price of a self-built system can break even in just a month or two of use. Read on to learn about the differences in building for RL vs. a typical deep learning box. This is the first on a series of posts on how we do meaningful work with reinforcement learning while saving $1000s on cloud compute bills.
</strong>
</blockquote>

# A Universe in a Box

<blockquote>
"If you wish to make an apple pie from scratch, first you must invent the universe." 
    
-Carl Sagan
</blockquote>

Carl Sagan reminded us that the entirety of human experience depends on our universe's exact set of physics, particularly amenable to giving rise to a warm, meaty species of ape that can build interesting things with their brains and bodies. Imagine any one of a number of physical constants or laws were different and life becomes unfathomable. We can consider a similar thought experiment in the context of developing novel forms of intelligence. Building rich simulated environments for machine agents to learn from and evolve in is one of the most promising routes for generating artificial general intelligence. If humans succeed in this venture before wiping themselves out, it will be a Big Deal, not least of all because it will help answer some of the Big Questions:

<blockquote>
<tt>
Human: Are there other intelligent beings in the universe beyond Earthlings?
    
AI: You're looking at one, Bub.
</tt>
</blockquote>

I've been building deep learning models since right after it was cool and a few years back I built a PC specifically for deep learning. The system was built around a pair of GTX 1060 6GB GPUs from Nvidia, and it massively accelerated my ability to iterate and experiment with image classification, segmentation, and pixel-to-pixel transformations, as well as machine learning for the optical inverse problems I work with professionally. A purpose-built system made a big difference in facilitating my learning curve, and this effect was amplified as I learned how to more effectively make use of both GPUs (and PyTorch and TensorFlow made it increasingly easy to train a model on multiple GPUs). I saved a lot of money by building my own system as compared to working through the same tutorials and experiments in GPU-enable cloud instances, and the PC performs well at its intended objective of training deep convolutional models, mainly for vision. But performance specificity comes with a tradeoff, and in the past 6 months or so I've found that my deep learning box was falling short at certain tasks, particularly evolutionary algorithms and reinforcment learning.

Aside from notebooks like Google's Colaboratory or Kaggle notebooks, I didn’t seriously give cloud computing a try until about the same time I started seriously studying reinforcement learning. A generous heap of Google Cloud Platform credits fell into my lap for competing in a NeurIPs 2019 reinforcement learning competition (Learning to Move Around). The goal of the competition was to learn a control policy for a biomechanical model of a human skeleton that could move toward a goal location. I found that while my system was probably overpowered with respect to running the control and value policy networks, it was spending most of the time computing physics for the reinforcement learning environment. This is typical for reinforcement learning, and consequently CPU performance is essential for fast learning in RL. Training on a measly 4 threads available on my budget CPU, my agent learned a rather pathetic local optima of face-planting enthusiastically in the general direction of the goal location<a href="#foot1">*</a> , which was nonetheless good enough for 10th place in the competition.

Using the credits from the competition, I began to familiarize myself with Google Compute Engine. With GCE, it's easy to spin up an instance with 32, 64, 96, or even more vCPUs. This makes a big difference if you have the right workload. Meanwhile, I was spending more time studying open source implementations of of RL and evolutionary algorithms and I came to appreciate the importance of multi-threaded execution for these types of algorithms, many of which are <a href="https://en.wikipedia.org/wiki/Embarrassingly_parallel">embarassingly parallel</a>. Simulation is a cornerstone of reinforcement learning and evolution methods, and efficiently simulating rich environments <em>in silico</em> comes with its own set of hardware considerations and trade-offs. Since simulation is a big part of reinforcement learning, building a workstation specifically for RL has more in common with a build for simulating physics or rendering workflows than a 'typical' deep learning box. Of course this also means the resulting PC will be great for turning the knobs all the way up on Kerbal Space Program. 

# Why RL? Why EA?

You're already familiar with the 2012 <a href="https://en.wikipedia.org/wiki/AlexNet">"ImageNet Moment"</a> when a GPU-trained convolutional neural network from Alex Krizhevsky, Ilya Sutstkever, and Geoffrey Hinton blew away the competition at the <a href="https://en.wikipedia.org/wiki/ImageNet_Large_Scale_Visual_Recognition_Challenge">ImageNet Large Scale Visual Recognition Challenge</a>. It was a big part of ushering in the (renascent) era of deep learning in general, but in particular it led to the widescale adoption of models and optimizers that could do pretty well on just about any image-based task. Transfer learning meant that a pre-trained network, even when trained on a vastly different task, could perform pretty well on a new task by replacing or re-initializing the parts of the network that put together higher-level abstractions and fine-tuning with a minimal amount of new training. Everything from image classification to semantic segmentation, and even caption generation can be amenable to this sort of transfer learning. In 2018 we saw the advent of very large, attention-based, transformer models for natural language processing, the impact of which has been strongly reminiscent of the original <a href="https://thegradient.pub/nlp-imagenet/">ImageNet Moment</a>. Just like transfer learning with image-based models, big transformers pre-trained on a large non-specifc text corpus can be readily applied to new problem domains. 

Even though reinforcement learning has seen major breakthroughs in the past decade, we are still arguably working toward a development at the scale and impact of AlexNet or the big transformer models. The AlphaGo lineage has decidedly disrupted the world of game-playing and machine learning in board games, and the latest iteration, MuZero, can play Go, chess, shogi, and Atari at a state-of-the-art level. But the model and policies learned by MuZero for chess is can't be used to improve its Go game or inspire strategies in Atari, let alone learning a new game or concepts about humans and game-playing in general. It's only in the last few years that "training on the test set" has gone from an acceptable practice and sheepish inside joke to being something to be ashamed of, thanks to better procedurally generated environments and richer simulations. We're starting to see legitimately interesting progress in exploration, generalization, and simulation-to-real learning transfer. There's also an ongoing resurgence in evolutionary algorithms that looks promising, especially when learning and evolution are combined. On top of all that, there's an incredible contest playing out between more general, (and compute efficient) open-ended learning algorithms and more structured (and sample efficient) approaches like inverse reinforcement learning and model-based methods. <em>En route</em> to understanding general intelligence, we can expect a slew of real-world applications in robotics control, electric grid management, and many other fields. I also predict fruitful cross-pollination between RL/EA research and physics simulations as hardware, software, and algorithmic development accompanies increased interest. I hope you'll tune in to future posts for discussions on the topics above, but in general let us agree that it's a great time to work on these areas. 

# Build, Buy, or Rent? 

If you want to run experiments with multicore workloads, the cost advantage of a physical workstation over cloud-based virtual machines is undeniable. I built a deep reinforcement learning system based on the 3rd generation Ryzen 3960X Threadripper which handily outperforms n1-highcpu instances with 64 vCPUs on Google Compute Engine in my benchmarks (coming soon to a subsequent post near you!). Based on this build, the cost comparison is shown below. 

<img src="/assets/build_a_rl_pc/gcp_vs_3960X.png">

The break-even point for the more expensive parts list below is about two months against on-demand Google Compute Engine instances (n1-high-cpu with 64 vCPUs). If you look at the figure, there's quite a big gap in the cost of running the GCE instances versus a purpose-driven self-built machine over the course of a year of use. That's more than $15,000.00. That might sound like an overestimate, but actually in my benchmarks the 3960X build is almost twice as fast as a n1-highcpu-64 instance, so it's _actually probably an underestimate._The figure is baed on usage time, not total time elapsed, so if you spend a significant amount of time iterating and building during the day, and regularly run experiments over nights and weekends, I would be surprised if you don't break even before 4 months and save thousands of dollars in the first year, wall time. 

Maybe the up-front cost of building at home is daunting, you are too nervous to risk bricking a high-end CPU that might cost more than a thousand dollars, or you just find the flexibility of cloud computing too appealing. Those are all valid reasons to pursue the cloud route instead, and if you do I suggest you spend some time optimizing for value efficiency. It's easy to lose track of the fact that you pay for storage, even if your instances are shut down, so keep an eye on those enormous datasets. Worried you might forget to shut down a big compute instance? You should probably <a href="https://cloud.google.com/compute/docs/shutdownscript">automate that.</a> You can also decrease cloud costs substantially by purchasing commited instances in advance (about 30% cheaper) or by renting pre-emptible instances (almost 5x cheaper) although those could be, well, empted at any time.
 

If your interest is piqued, and I hope that it is, I've put together a couple of parts lists taking advantage of the latest generation of Ryzen CPUs from AMD. There's a big budget option at about $3000(it's the build I went with) and a medium-big budget option at about $1000 less. If you have more cash burning a hole in your pocket you can also spend $500 to $1500 dollars more on the big budget option by upgrading to a 32-core Threadripper 3970X or a massive 64-core Threadripper 3990X. But keep in mind that <a href="https://en.wikipedia.org/wiki/Amdahl%27s_law">Amdahl's law</a> prevents parallelization speedups from scaling linearly, so the 3990X won't be twice as fast as the 3970X and the 3960X is the best performance-per-dollar of the new sTRX4 Threadripper chips. 

# Picking Parts 

## Souped-up, yet sane

| Component Type | Part | Price | 
|:------------------|:--------------------------------:|-----------------------:|
|**CPU:** | AMD Ryzen Threadripper 3960X 3.8 GHz 24-core TRX4 processor | \$1499.99|
|**CPU Cooler** | be quiet! Dark Rock Pro TR4 CPU air cooler | \$89.90|
|**Motherboard:** | ASRock TRX40 Creator TRX4 ATX motherboard | \$459.99|
|**Memory:** | Ballistix 64GB Sport LT Series DDR4 3200 MHz | \$313.99|
|**Storage:** | Samsung 1TB 970 EVO NVMe M.2 internal SSD | \$169.99|
|**Video Card:** | Gigabyte GeForce GTX 1060 6GB 6 GB WINDFORCE OC 6G Video Card | ~\$210.00|
|**PSU:** | SeaSonic FOCUS Plus Platinum 850 Watt 80+ Platinum Fully Modular ATX Power Supply| \$154.99 |
|**Case:** | be quiet! Silent Base 601 Mid-Tower ATX Case | \$129.00 |
|**Monitor:** | Dell S2419HN 23.8" 1920x1080 60 Hz Monitor | \$159.79 |
|**Keyboard** | Das Keyboard Model S Professional Mechanical Keyboard (Gamma Zulu switches)<a f="#foot2">\*\*</a> | \$119.00 |
|**Mouse:** | Logitech Trackman Marble| \$29.99 |
|**__Total:__** | | **__\$3336.63__** |
|**__Headless Total:__** | | **__\$3027.85__** |

<div align="center"><a href="https://pcpartpicker.com/list/sczgp8">Parts list on PCPartpicker</a></div>

## Baby Threadripper 3950x

| Component Type | Part | Price | 
|:------------------|:--------------------------------:|-----------------------:|
|**CPU:** | AMD Ryzen 9 3950X 3.5 GHz 16-Core Processor | \$738.89|
|**CPU Cooler** | be quiet! Dark Rock Pro TR4 CPU air cooler | \$86.46|
|**Motherboard:** | ASRock TRX40 Creator TRX4 ATX motherboard | \$259.00|
|**Memory:** | Ballistix 64GB Sport LT Series DDR4 3200 MHz | \$313.99|
|**Storage:** | Samsung 1TB 970 EVO NVMe M.2 internal SSD | \$169.99|
|**Video Card:** | Gigabyte GeForce GTX 1060 6GB 6 GB WINDFORCE OC 6G Video Card | ~\$210.00|
|**PSU:** | SeaSonic FOCUS Plus Platinum 850 Watt 80+ Platinum Fully Modular ATX Power Supply| \$154.99 |
|**Case:** | be quiet! Silent Base 601 Mid-Tower ATX Case | \$129.00 |
|**Monitor:** | Dell S2419HN 23.8" 1920x1080 60 Hz Monitor | \$159.79 |
|**Keyboard** | Das Keyboard Model S Professional Mechanical Keyboard (Gamma Zulu switches)<a f="#foot2">\*\*</a> | \$119.00 |
|**Mouse:** | Logitech Trackman Marble| \$29.99 |
|**__Total:__** | | **__\$2371.10__** |
|**__Headless Total:__** | | **__\$2062.32__** |

<div align="center"><a href="https://pcpartpicker.com/list/KhWdgJ">Parts list on PCPartpicker</a></div>

<a name="foot1">\*</a> In the end I settled on training with the Spinning Up repository from Joshua Achiam at OpenAI. This is a great resource (now with PyTorch implementations) and I highly recommended playing around with it. 
<a href="#foot1_origin">back to text</a>

(although MuZero still falls short at RL hard Atari games like Montezuma's revenge).
