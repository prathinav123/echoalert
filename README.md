# EchoAlert

A tool that listens to a room and tells you what it hears, built for people who can't rely on their ears to catch important sounds.

## Who this is for

This project is built for people living with hearing loss, especially anyone dealing with Sudden Sensorineural Hearing Loss (SSHL). SSHL is exactly what it sounds like: hearing loss that shows up fast, sometimes overnight, often in just one ear, and often with no clear cause. Someone can go to bed with normal hearing and wake up unable to hear out of one side at all.

But this isn't only for sudden cases. Anyone with partial hearing loss, single-sided deafness, or hearing that's declined over time faces the same daily problem: a lot of important sounds just don't register. A smoke alarm going off in another room. A doorbell. Someone calling your name from the kitchen. For most people these sounds are automatic. You hear a knock, you know it's a knock, you go answer the door. For someone who's lost part of their hearing, that whole chain can break, and the stakes aren't always small. Missing a smoke alarm is a real safety problem, not just an inconvenience.

Most solutions to this right now are either expensive (hearing aids with built-in alert features) or pretty limited (a vibrating alarm that buzzes but doesn't tell you what set it off). EchoAlert is an attempt at something in between: free, software-based, and specific about what it heard.

## What EchoAlert actually does

There are two halves to this project.

**Half one: the listener.** A Python program that uses a microphone to listen to a room in real time, recognizes specific sounds (doorbells, alarms, knocking, sirens, speech), and immediately shows an alert when it hears one.

**Half two: the analyst.** Once the listener has been running for a while, it's been quietly logging everything it hears into a database. The second half of this project goes back through that log and asks: how good is this thing actually at its job? When does it get fooled? Does a noisy room throw it off? This half turns raw detection logs into an honest report card for the system.

## The tech, explained without the jargon

### YAMNet: the part that "hears"

The obvious way to build something like this would be to train your own sound-recognition AI from scratch. That takes enormous amounts of recorded audio and serious computing power, neither of which this project has.

Instead, EchoAlert uses YAMNet, a model Google already built and gave away for free. YAMNet has been trained on a huge number of real-world sounds and can recognize about 500 different categories out of the box: dogs barking, glass breaking, footsteps, engines, doorbells, alarms, and a lot more. The actual engineering work here isn't teaching a computer what a doorbell sounds like. Google already did that. The work is everything around it: getting live microphone audio into YAMNet fast enough to be useful, figuring out which of its 500 categories actually matter for this project, and deciding when a detection is confident enough to be worth an alert.

### Turning sound into numbers

A computer can't listen the way a person does. Before YAMNet can recognize anything, a sound has to be turned into numbers it can work with. This happens through a few ideas:

- **Sample rate**: how many times per second a microphone measures the sound wave. YAMNet expects exactly 16,000 measurements per second. Feed it audio recorded at a different rate and it won't work right, which is a lesson this project already ran into firsthand.
- **Waveform**: a plain plot of loudness over time. Nothing fancy, just how strong the sound pressure is at each instant.
- **Spectrogram**: instead of just loudness, this shows which frequencies are present at each moment, kind of like a fingerprint of a sound. A doorbell chime and a dog bark look completely different once you can see their frequency pattern, even if their volume looks similar on a plain waveform.

### Confidence scores and why averaging can be misleading

Every time YAMNet looks at a chunk of audio, it doesn't say "this is definitely a doorbell." It gives a confidence score for every one of its 500 categories, something like 0.95 for "Doorbell" and 0.02 for "Dog bark" and so on down the list. The project has to decide how high that number needs to be before it counts as a real detection, because set the bar too low and you get alerts for things that never happened, set it too high and you miss real ones.

One thing this project already learned the hard way: a chunk of audio that's mostly silence, with a doorbell ring buried in just one second of it, can end up with "Silence" as the average top prediction, even though the doorbell was detected clearly for that one second. Averaging across a whole clip can hide a real detection. Looking frame by frame, in small slices of under a second each, turned out to be the more honest way to see what YAMNet actually noticed.

### Filtering down to what matters

YAMNet knows about 500 sounds, and this project cares about maybe half a dozen: doorbells, alarms, knocking, sirens, speech, maybe a dog bark. Part of the work is mapping YAMNet's specific labels down to those categories that actually matter for someone with hearing loss. This turned out to be less obvious than it sounds. An alarm clock's beeping, for instance, doesn't always get labeled "Alarm" by YAMNet. Sometimes it comes back as "Telephone" or "Ringtone" or "Beep, bleep," because those sounds are all electronically similar. So instead of watching for one exact label, the system has to watch for a small group of related labels that all point toward the same real-world event.

### Storing what it hears

Every time EchoAlert detects something worth alerting on, it writes that down: what it heard, when, and how confident it was. This goes into a SQLite database, which is really just a small file that stores organized tables of information, similar to a spreadsheet but built for a computer to search through quickly. This step matters because it means the system's performance isn't something you have to just take on faith. Every detection is on record, which means it can be checked against reality later.

### Studying its own performance

This is where the second half of the project comes in. With a database full of logged detections, it becomes possible to ask real questions using SQL, a language built for searching and summarizing tables of data: Which sounds get detected most often? Does confidence drop at certain times of day? How often does it fire an alert for something that wasn't actually there?

Two ideas matter most here:

- **Precision**: out of everything the system flagged as a doorbell, how many actually were a doorbell? Low precision means a lot of false alarms.
- **Recall**: out of every doorbell that actually rang, how many did the system catch? Low recall means it's missing real events.

A system could have great precision by barely ever guessing "doorbell," and terrible recall as a result. Both numbers matter, and there's a real trade-off between them.

### Showing it all on screen

None of this is useful if it just sits in a database. Streamlit turns a plain Python script into a simple website, without needing to write any HTML or JavaScript. That's what powers the live dashboard: a screen that shows what EchoAlert is currently hearing, pulled straight from the same database everything gets logged to.

## Why this matters

The point isn't to build something flashy. It's to take a real, specific accessibility problem, solve it using tools that already exist and are free, and then actually check whether the solution works, instead of just assuming it does. That second part, the honest evaluation, is what turns this from a demo into something closer to a real tool.