-----------------------------------------------------------------------

-- Define loss function 
-- Here we use LogSoftMax function
-- Our Criterion is negative loglikelihood criterion

-----------------------------------------------------------------------

print '==> define loss'

criterion = nn.ClassNLLCriterion()

-----------------------------------------------------------------------

print '==> here is the loss function:'
print(criterion)
