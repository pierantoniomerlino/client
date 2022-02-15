package triton.client.examples;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import com.google.common.collect.Lists;

import jep.Interpreter;
import jep.NDArray;
import jep.SharedInterpreter;
import triton.client.InferInput;
import triton.client.InferRequestedOutput;
import triton.client.InferResult;
import triton.client.InferenceException;
import triton.client.InferenceServerClient;
import triton.client.pojo.DataType;

public class ImageClient {

    public static void main(String[] args) {
        if (args == null || args.length == 0) {
            log("Missing image path");
            System.exit(1);
        }
        String path = args[0];

        InferenceServerClient client = null;
        try {
            client = new InferenceServerClient("0.0.0.0:4000", 5000, 5000);
            boolean isBinary = true;
            imageInference(client, path, isBinary);
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

    private static void imageInference(InferenceServerClient client, String path, boolean isBinary)
            throws InferenceException {
        NDArray<double[]> preprocessedInputData = (NDArray<double[]>) preprocess(path);
        float[] preprocessedInputDataConverted = new float[preprocessedInputData.getData().length];
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < preprocessedInputData.getData().length; i++) {
            preprocessedInputDataConverted[i] = (float) preprocessedInputData.getData()[i];
            sb.append(preprocessedInputDataConverted[i]).append(",");
        }
        log(sb.toString());

        String modelName = "densenet_onnx";
        InferInput input = new InferInput("data_0", new long[] { 3L, 224L, 224L }, DataType.FP32);
        input.setData(preprocessedInputDataConverted, isBinary);

        List<InferInput> inputs = Lists.newArrayList(input);
        List<InferRequestedOutput> outputs = Lists.newArrayList(new InferRequestedOutput("fc6_1", isBinary, 1));

        InferResult inferResult = client.infer(modelName, inputs, outputs);
        List<String> results = postprocess(inferResult);

        results.forEach(ImageClient::log);
    }

    private static Object preprocess(String path) {
        try (Interpreter interp = new SharedInterpreter()) {
            interp.exec("from PIL import Image");
            interp.exec("import preprocessor");
            interp.set("filename", path);
            interp.exec("img = Image.open(filename)");
            Object img = interp.getValue("img");
            interp.exec("pre = preprocessor.preprocess");
            Object result = interp.invoke("pre", img, DataType.FP32, 0, 224, 224, "INCEPTION");

            return result;
        }

    }

    private static List<String> postprocess(InferResult result) {
        System.out.println(result.getOutputs());
        String results = new String(result.getOutputAsByte("fc6_1"));
        return Arrays.asList(results.split(":"));
    }

    private static void log(String message) {
        System.out.println(message);
    }

    private static void log(String message, Exception e) {
        System.out.println(message + " " + e.getLocalizedMessage());
    }
}

// name: "densenet_onnx"
// platform: "onnxruntime_onnx"
// max_batch_size : 0
// input [
// {
// name: "data_0"
// data_type: TYPE_FP32
// format: FORMAT_NCHW
// dims: [ 3, 224, 224 ]
// reshape { shape: [ 1, 3, 224, 224 ] }
// }
// ]
// output [
// {
// name: "fc6_1"
// data_type: TYPE_FP32
// dims: [ 1000 ]
// reshape { shape: [ 1, 1000, 1, 1 ] }
// label_filename: "densenet_labels.txt"
// }
// ]
