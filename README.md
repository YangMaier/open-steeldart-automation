# opendarts-steeldart-recognition
This repository is part of my Bachelor's Thesis.

I will organise the work packages in this [Trello Board](https://trello.com/b/ET7HVk3v/steeldart-recognition).

While I am working on the thesis, I will not accept any merge requests or suggestions. Until it is finished, this will not be a project that can be used to automatically read live dart games.

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

There are multiple goals i want to complete:
- [ ] Create a minimal working solution with OpenCV and Python with uncalibrated webcams
- [ ] Calibrate the cameras via the standardized size of the dartboard
- [ ] Implement a depths promotion network to generate test data for a Conditional Random Field (CRF)
- [ ] Train a CRF to predict scores based on darts that still fly
- [ ] Train a net for dartboard segmentation
- [ ] Train a net for dart impact point detection
- [ ] Build a point score proposal algorithm to use with multiple webcams
- [ ] Build a test suite to test different solutions
- [ ] Determine the effects of: 1. multiple webcams, 2. impact point prediction, 3. a combination of solutions; on scoring precision and reliabilty
