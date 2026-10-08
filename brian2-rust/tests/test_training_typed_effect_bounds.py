"""Keep actual NumPy array casting failures explicit at conversion."""
import brian2 as b
import pytest
from brian2_rust import lower_brian_dynamic_training,TrainingConversionError
from test_training_typed_callback_effects import model

def fraction(x):
 x*=.8
 return x

def subtract(x):
 x-=True
 return x

def negative(x):
 x*=True
 return -x

@pytest.mark.parametrize('callback,field,message',[(fraction,'k','dtype'),(subtract,'flag','boolean subtraction'),(negative,'flag','unary arithmetic')])
@pytest.mark.parametrize('discard',[False,True])
def test_original_numpy_failures_are_refused_before_native_launch(callback,field,message,discard):
 net,g,dt,_=model('borrowed',discard,None,None,'cpu')
 runner=next(obj for obj in g.contained_objects if obj.name.endswith('run_regularly'))
 runner.abstract_code='v=change('+field+')'
 g.namespace['change']=b.Function(callback,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 source=next(obj for obj in net.objects if obj.name=='typed_effect_input');hidden=next(obj for obj in net.objects if obj.name=='typed_effect_hidden')
 with pytest.raises(TrainingConversionError,match=message):lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g])
 with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})
