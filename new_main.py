在pytroch 0.4及其以后的版本中，用torch.no_grad()这个上下文管理器就可以了，例子如下：


model.train()
# here train the model, just skip the codes
model.eval() # here we start to evaluate the model
with torch.no_grad():
	for each in eval_data:
		data, label = each
		logit = model(data)
		... # here we just skip the codes


如上，我们只需要在加上上下文管理器就可以很方便的取消掉梯度。
这个功能在pytorch以前的版本中，通过设置volatile=True生效，不过现在这个用法已经被抛弃了。