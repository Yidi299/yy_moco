optimizer_sam = optimizer
optimizer = optimizer_sam.base_optimize

loss = criterion(output, target)
# first forward-backward pass
optimizer.zero_grad()
loss.backward()
# optimizer.synchronize()
optimizer_sam.first_step(zero_grad=True)
# second forward-backward pass
output2 = model(images)
loss2 = criterion(output2, target)
if math.isnan(loss2):
    raise RuntimeError("ERROR: Got NaN loss2")
loss2.backward()
optimizer.synchronize()
with optimizer.skip_synchronize():
    optimizer_sam.second_step()                    