package triton.client.examples;

import java.io.IOException;
import java.util.List;

import com.google.common.collect.Lists;

import triton.client.InferInput;
import triton.client.InferRequestedOutput;
import triton.client.InferResult;
import triton.client.InferenceException;
import triton.client.InferenceServerClient;
import triton.client.pojo.DataType;

public class SimpleIrisInferClient {

    public static void main(String[] args) {
        InferenceServerClient client = null;
        try {
            client = new InferenceServerClient("0.0.0.0:8000", 5000, 5000);
            boolean isBinary = true;
            float[] irisData = { 5.4F, 3.7F, 1.5F, 0.2F };
            irisInference(client, irisData, isBinary);
        } catch (IOException | InferenceException e) {
            log("Error", e);
        } finally {
            if (client != null) {
                try {
                    client.close();
                } catch (Exception e) {
                    log("Error", e);
                }
            }
        }
    }

    private static void irisInference(InferenceServerClient client, float[] irisData, boolean isBinary)
            throws InferenceException {
        String modelName = "simple_iris";

        InferInput input = new InferInput("fc1_input", new long[] { 1, 4 }, DataType.FP32);
        input.setData(irisData, isBinary);

        List<InferInput> inputs = Lists.newArrayList(input);
        List<InferRequestedOutput> outputs = Lists.newArrayList(new InferRequestedOutput("output", isBinary));

        InferResult result = client.infer(modelName, inputs, outputs);

        float[] output = result.getOutputAsFloat("output");
        for (int i = 0; i < output.length; i++) {
            log(String.valueOf(output[i]));
        }
    }

    private static void log(String message) {
        System.out.println(message);
    }

    private static void log(String message, Exception e) {
        System.out.println(message + " " + e.getLocalizedMessage());
    }

}
