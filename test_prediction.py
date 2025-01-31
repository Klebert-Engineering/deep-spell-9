# (C) 2018-present Klebert Engineering

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.models.extrapolator import DSLstmExtrapolator
from deepspell.models.discriminator import DSLstmDiscriminator

def test_models():
    print("Loading models...")
    try:
        # Load extrapolator model
        extrapolator_model = DSLstmExtrapolator("models/deepsp_extra-v2_na_lr003_dec50_bat3192_128-128-128.json")
        print("✓ Extrapolator model loaded successfully")
        
        # Load discriminator model
        discriminator_model = DSLstmDiscriminator("models/deepsp_discr-v3_na-lower_lr003_dec50_bat3072_fw128-128_bw128.json")
        print("✓ Discriminator model loaded successfully")
        
        # Verify featureset compatibility
        assert extrapolator_model.featureset.is_compatible(discriminator_model.featureset)
        print("✓ Featuresets are compatible")
        
        # Test discrimination
        test_input = "Los Angeles"
        print(f"\nTesting discrimination with input: '{test_input}'")
        completion_classes = discriminator_model.discriminate(extrapolator_model.featureset, test_input)
        print("Discrimination results:")
        for i, classes in enumerate(completion_classes):
            if i < len(test_input):
                print(f"Character '{test_input[i]}': {classes[:3]}")
            else:
                print(f"EOL prediction: {classes[:3]}")
            
        # Test extrapolation
        test_prefix = "Los Angeles"
        print(f"\nTesting extrapolation with prefix: '{test_prefix}'")
        # Get class names from discriminator
        prefix_class_names = [col[0][0] for col in discriminator_model.discriminate(extrapolator_model.featureset, test_prefix)][:-1]
        print("Detected classes:", prefix_class_names)
        
        completions = extrapolator_model.extrapolate(
            extrapolator_model.featureset,
            test_prefix,
            prefix_class_names,
            16)  # predict up to 16 chars
            
        print("Extrapolation results:")
        for completion in completions:
            print(completion)
            
        print("\n✓ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_models() 