---
layout: post
title: "Evaluating Open-Ended Exploration and Creation" 
date: 2021-02-12 00:00:00 +0000
categories: cellular_automata carle
---

<br>
<div align="center">
<img width="100%" src="/assets/carle/use_this.png">
</div>
<br>

# Challenges in evaluating open-ended exploration and creation

I've been working on a reinforcement learning (RL) environment for machine exploration and creativity using Life-like cellular automata. Called CARLE (for Cellular Automata Reinforcement Learning Environment), the environment is the basis for an [official competition](https://ieee-cog.org/2021/index.html#competitions_section) at the third IEEE Conference on Games. But Carle's Game is somewhat unusual in that it is a challenge in open-endedness. In fact, the RL environment doesn't even have a native reward value, although there are several exploration reward wrappers available in the repository. 

At the risk of pointing out the obvious, judging a contest with no clear goal is a challenge in and of itself. However, in my humble opinion, this is the type of approach that is most likely to yield the most interesting advances in artificial intelligence. Qualitative advances in a vast universe are what drives progress in humanity's humanity*, even though judging these types of outputs are more difficult than recognizing an improvement in the existing state-of-the-art. The difficulty in recognizing qualitative advances contributes to the ambiguity of trying to evaluate works of art or art movements. 

The first transistors did not improve on the performance of the mechanical and vacuum tube-based computers available at the time any more than the quantum computers in use today can unequivocally outperform electronic computers on useful tasks. Likewise, improvements in AI benchmarks like ImageNet or the Atari suite do not intrinsically bring us closer to general AI. 

Life-like CA have generated a rich ecosystem and many exciting discoveries by hobbyists and professional researchers alike. Carle's Game is intended to reward both artistic creativity and ingenuity in machine agents, and in this post I tinker with several rulesets to learn something about my own preferences and what determines whether
CA outputs are interesting or boring from my own perspective. 

<em>* We'll probably need a more inclusive term for what I'm trying to express when I use this word here, although it's impossible to predict the timeline for machine agents claiming personhood and whether or not they might care about semantics.</em>

## Arts and crafting

Life-like CA can produce a wide variety of machines and pleasing forms. Some rulesets seem to produce pleasing patterns intrinsically, akin to the natural beauty we see in our own universe in planets and nebulas, rocks and waves, and other inanimate phenomena. This will be our starting point for determining contributes to an interesting output. 

In the following comparison of a scene generated with random inputs for two different rulesets, which one is more interesting? 

<br>
<div align="center">
<img width="80%" src="/assets/carle/boring_vs_cool.png">
</div>
<br>
You may want to zoom in to get a good feel for each output, although aliasing of small features in a scaled down render can be interesting in their own right. 

If you have similar aesthetic preferences as I do, you probably think the image on the right is more interesting. The output on the right (generated with the "Maze" ruleset) has repeating motifs, unusual tendril-like patterns, borders, and a ladder-like protrusion that looks interesting juxtaposed against the more natural-looking shapes prevalent in the rest of the pattern. The left image, on the other hand, looks more like a uniform and diffuse cloud centered on a bright orb (corresponding to the action space of the environment). One way to rationalize my own preference for the pattern on the right is that it contains more surprises, while simultaneously appearing more orderly and complex. 

The "boring" ruleset, at least when displayed in the manner above by accumulating cell states over time, is known as 34-Life and has birth/survive rules that can be written as B34/S34 in the language of Life-like CA rules. The more interesting CA is unsurprisingly call Maze and has rules B3/S12345. 

Here's a pattern produced by another ruleset with some similarities to Maze:

<br>
<div align="center">
<img width="80%" src="/assets/carle/coral_accumulation.jpg">
</div>
<br>

That image was generated with a modified Coral ruleset, aka B3/S345678. In my opinion this ruleset demonstrates a substantial amount of natural beauty, but we can't really judge a creative output by what essentially comes down to photogenic physics. That's a little bit like if I were to carry a large frame with me on a hike and use it to frame a nice view of a snowy mountain, then sitting back to enjoy the natural scene while smugly muttering to myself "that's a nice art." To be honest now that I've written that last sentence it sounds really enjoyable. 

There's an interesting feature in the coral-ish ruleset image, one that contrasts nicely with the more biological looking patterns that dominate. A number of rigid straight features propagate throughout the piece, sometimes colliding with other features and changing behavior. It looks mechanical, and you might feel it evokes a feeling of an accidental machine, like finding a perfect staircase assembled out of basalt.

<br>
<div align="center">
<img width="80%" src="/assets/carle/giants_causeway.jpg">
<br>
<em>Giant's Causeway in Northern Ireland. Image CC BY SA Wikipedia user <a href="https://commons.wikimedia.org/wiki/User:Sebd">Sebd</a></em>
</div>
<br>

Regular formations like that are more common than one might naively expect (if you've never seen a nature before), and throughout history interesting structures like Giant's Causeway have attracted mythological explanations. If you were previously unaware of the concept of stairs and stumbled across this rock formation, you might get a great idea for connecting the top floors to the lower levels of your house. Likewise, we can observe the ladder-like formations sometimes generated by the modified Coral ruleset and try to replicate it, and we might want to reward a creative machine agent for doing something similar. If we look at the root of the structure, we can get an idea of how it starts and with some trial and error we can find a seed for the structure. 

<br>
<div align="center">
<img width="80%" src="/assets/carle/coral_ladder_seed.png">
<br>
<em>Coral ladder seed. We'll pay some homage to the story of John H. Conway coming up with the Game of Life on a breakroom Go board to illustrate machines in Life-like CA. </em>
</div>
<br>

When subjected to updates according to the modified Coral ruleset, we see the ladder-like structure being built. 

<br>
<div align="center">
<img width="80%" src="/assets/carle/coral_ladder.gif">
</div>
<br>

Although Coral and the modified ruleset shown here is very different from John Conway's Game of Life, we can relate this ladder-like structure to a class of phenomena found in various Life-like CA: gliders and spaceships. A glider is a type of machine that can be built in Life-like CA that persists and appears to travel across the CA universe. These can be extremely simple, and make good building blocks for more complicated machines. In Life, a simple glider can be instantiated as in the figure below.

<br>
<div align="center">
<img width="80%" src="/assets/carle/life_glider.png">
<img width="80%" src="/assets/carle/life_glider.gif">
</div>
<br>

Spaceships are like gliders, and in general we can think of them as just another name for gliders that tend to be a bit larger. The space of known spaceships/gliders in Life is quite complex, and they vary substantially in size and even speed. Support for gliders in a given CA universe also tells us something about the [class of a CA ruleset](https://www.ics.uci.edu/~eppstein/ca/), which has implications for A CA's capabilities for universal computation. Searching for gliders in CA has attracted [significant](https://arxiv.org/abs/cs/0004003) [effort](https://www.researchgate.net/publication/221220296_New_approach_to_search_for_gliders_in_cellular_automata) [over](https://www.researchgate.net/publication/221024195_Searching_for_Glider_Guns_in_Cellular_Automata_Exploring_Evolutionary_and_Other_Techniques) the [years](https://www.researchgate.net/publication/224570926_Genetic_Approaches_to_Search_for_Computing_Patterns_in_Cellular_Automata), and gives us some ideas for how we might evaluate curious machine agents interacting with Life-like CA. We can simply build an evaluation algorithm that computes the mean displacement of the center of mass of all live cells in a CA universe and how it changes over a given number of CA timesteps. Although this does give an advantage faster gliders, which are not necessarily more interesting, it provides a good starting point for developing creative machine agents that can learn to build motile machines in arbitrary CA universes. Clearly it wouldn't make sense to compare the same quantitative value of that metric for a Coral ladder versus a Life glider, but we could evaluate a set suite of different CA universes to get an overall view of agents' creative machinations.

## What can a machine know about art, anyway?

Evaluating agent performance with an eye toward rewarding gliders is one way to evaluate Carle's Game, but if that's all we did we'd be constricting the challenge so severely it starts to look like a more standard benchmark, and it would no longer provide a good substrate for studying complexity and open-endedness. I would also like to encourage people from areas outside of a conventional machine learning background to contribute, and so in addition to rewarding agents that discover interesting machines, we should also try to reward interesting and beautiful artistic expression. 

We can consider using automated means to evaluate agent-CA interactions based on quantitative preconceptions of art and beauty, for example by rewarding different types of symmetry. Or we could use [novelty-based reward functions](https://rivesunder.gitlab.io/rl/2019/08/24/breaking_rnd.html) like [random network distillation](https://www.semanticscholar.org/paper/Exploration-by-Random-Network-Distillation-Burda-Edwards/4cb3fd057949624aa4f0bbe7a6dcc8777ff04758) or autoencoder loss. 

We can also try to reward machine creations for the impact they have on a human audience. In the final submission evaluation for the contest at IEEE CoG, I plan to incorporate a people's choice score as well as to solicit curation from human judges that have expertise in certain areas. But soliciting human judgement from the crowd and experts while the contest was underway would not only require a prohibitive amount of human effort, it could change the final outcome. An online audience voting for the best creations might grow bored with interesting output prematurely, and competing teams might borrow inspiration from others or outright steal patterns as they are revealed in the voting interface. 

Instead of online voting during competition, I am considering training an "ArtBot" value model that can provide stable feedback while the competition is still going. I'm still working out what this will entail, but I plan to open the competition beta round in March with the aim of eliciting feedback and pinning down competition processes. It might be as simple as training a large conv-net on image searches like ["not art"](https://duckduckgo.com/?q=%22not+art%22&t=brave&iax=images&ia=images) or ["good generative art"](https://duckduckgo.com/?q=good+generative+art&t=brave&iar=images&iax=images&ia=images), but we can probably expect better results if we take some ideas from the [picbreeder](http://picbreeder.org/) project. Picbreeder is a website and project led by Jimmy Secretan and Kenneth Stanley where users select the best image from a few options, which are then evolved using ideas like [NEAT](http://www.cs.ucf.edu/~kstanley/neat.html) and compositional pattern producing networks ([pdf](http://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf)). The results are quite impressive, and you can [view](http://picbreeder.org/search/showcategory.php?visited=151) the [results](http://picbreeder.org/search/showcategory.php?visited=103) on the website.

The culmination of the challenge will involve a final submission of creative agents, which will be presented with a secret suite of CA universes that will include but won't be limited to Life-like CA based on Moore neighborhoods. The test suite may also include CA from the larger generalized space of possible CA known as ["Larger than Life"](https://www.sciencedirect.com/science/article/abs/pii/S0167278903001556?via%3Dihub), which I think should provide enough diversity to make it difficult to game or overfit the evaluation while still being tractable enough to yield interesting results. 

If you're thinking of entering Carle's Game for IEEE CoG 2021 and/or have ideas about evaluating machine agents in open-ended environments, @ me on Twitter [@RiveSunder](https://twitter.com/rivesunder) or leave an issue on the GitHub [repository](https://github.com/rivesunder/carle) for CARLE. I look forward to meeting the creative machines you develop and discover. 

<br>
<div align="center">
<img src="/assets/carle/cool_b3s23.png" width="100%">
</div>
<br>

<em>Update 2021-02-14: Coral ladders are not found in the Coral universe, but rather can occur in rulesets between "Coral" (B3/45678) and "Life Withour Death" (B3/S012345678). In an earlier version of this post I describe finding the phenomenon in the Coral ruleset, but due to an implementation error I was actually working with B3/S345678. The text is updated to reflect the proper rules. </em>


