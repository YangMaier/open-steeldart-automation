# open-steeldart-recognition
This repository is part of my Bachelor's Thesis.

I will organise the work packages in this [Trello Board](https://trello.com/b/ET7HVk3v/steeldart-recognition).

While I am working on the thesis, I will not accept any merge requests or suggestions. Until the BA is finished, this will not be a project that can be used to automatically read live dart games.

Several steps are necessary to calculate the score of one or multiple darts in a steeldart board.
- Get the segments of dartboard fields
- Relate the ring figures to the segments
- Detect darts and their impact points
- Calculate the score for each dart. There must be a value that indicates how reliable and precise the result is. A confidence indicator.

There are various implementation options for each individual step.
- You can use one or more cameras
- A basic solution can be implemented without any neural networks or flight prediction
- A more sophisticated solution can be implemented with dart flight prediction
- Multiple neural networks can be trained, one for each necessary step of score calculation.

But which solution can shine in every szenario?

There are a number of cases that can be difficult to deal with. Lightning changes, dart flights drop off, etc..

Which solution is the most precise and reliable?

Is it really necessary to use more than one camera when the software solution is sophisticated and mature enough?

There are multiple goals i want to complete during the Bachelor's Thesis:
- [ ] Create a minimal working solution with OpenCV and Python with uncalibrated webcams
- [ ] Calibrate the cameras via the standardized size of the dartboard
- [ ] Implement a depths promotion network to generate test data for a Conditional Random Field (CRF)
- [ ] Train a CRF to predict scores based on darts that still fly
- [ ] Train a net for dartboard segmentation
- [ ] Train a net for dart impact point detection
- [ ] Build a point score proposal algorithm to use with multiple webcams
- [ ] Build a test suite to test different solutions
- [ ] Determine the effects of: 1. multiple webcams, 2. impact point prediction, 3. a combination of solutions; on scoring precision and reliabilty


I want this code to be OpenSource.

The Darts community is very nice, but all sophisticated, automatic dart recognition software solutions that are working well are ClosedSource at the time of writing.

And i hate that.

After i completed the thesis, i want to use this code to create a complete and OpenSource solution to steeldart recognition.

Minimum features will include (hopefully i have time for all that, oof.)
- [ ] a complete offline solution (backend, frontend, database, user administation)
- [ ] a frontend that can be shown in an app, not sure yet which techstack
- [ ] a headless online backend
- [ ] an online score and user database with login (that i will pay for, not sure yet which techstack)
- [ ] backend camera configuration sharing for better initial values on new setups
- [ ] an uncomplicated and structured way to add game modes
- [ ] the possibility to upload training data automatically (empty dartboards for field segmentation and ring figure recognition, boards with darts and their corresponding scores, picture sequence with flying dart to impact point for CRF training)
