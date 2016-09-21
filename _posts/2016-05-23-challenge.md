---
layout: post
comments: true
title: A spike train challenge
description: Recently, while discussing during a journal club, I wondered how easily a computational neuroscientist can distinguish between a real neuronal recording and a simulated dataset. So I took a real dataset, imitated it with a model, and I'm asking you to tell me which one is which.
imgurl: /files/challenge-timeseries.png
categories:
- blog
---

While working in computational neuroscience, I have faced two different, although related aspects:

- Understanding neural recordings, trying to guess elements of how the neurons interact with each other
or how the network encodes and processes information (data analysis)

- Building models of spiking neural networks, trying to be as accurate as possible in reproducing certain features.

Recently, while discussing during a journal club, I wondered how easily a computational neuroscientist
can distinguish between a real neuronal recording and a simulated dataset. Obviously, a model could
get arbitrarily close to a recording by simply copying it, or machine-learning its features.
So I took a biological recording, from a mouse retina receiving a periodic full-field stimulus
(light turned on and off alternately);
from it, I created a simulated dataset with the following constraints: I chose a model (a very simple, non-leaky,
binary integrate and fire), and I allowed myself to tweak its parameters and all the properties of the
stimulus, in order to match the biological dataset qualitatively, but only *manually*. No machine learning,
only myself experimenting with different values of the parameters until it "looked good".


So [here](/files/spiketrains.tar.gz) are the datasets, in matlab and numpy formats.
In the files, the first column contains the
spike times in seconds, the second contains the id of the neuron that
spiked (from 0 to 19). Here's a sample (raster and instantaneous activity):

<center>
<img src="/files/challenge-timeseries.png" width="550"/>
</center>

Can you tell me which one is which and why?
Email me your answers at mar-tino.sorba-ro@ed.ac.uk (removing the hyphens).
