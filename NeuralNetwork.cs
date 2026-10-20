class NeuralNetwork
{
    private int numInput;
    private int numHidden;
    private int numOutput;

    private double[] inputs;
    private double[][] ihWeights; // input hidden weights
    private double[] hBiases; // input biases
    private double[] hOutputs; // hidden outputs


    private double[][] hoWeights; // hidden output weights
    private double[] oBiases; // output biases
    private double[] outputs; // final outputs

    private Random rnd;

    private int numWeights
    {
        get
        {
            return (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
        }
    }

    public double[] GetWeights()
    {
      double[] result = new double[numWeights];
      int k = 0;
      for (int i = 0; i < ihWeights.Length; ++i)
        for (int j = 0; j < ihWeights[0].Length; ++j)
          result[k++] = ihWeights[i][j];
      for (int i = 0; i < hBiases.Length; ++i)
        result[k++] = hBiases[i];
      for (int i = 0; i < hoWeights.Length; ++i)
        for (int j = 0; j < hoWeights[0].Length; ++j)
          result[k++] = hoWeights[i][j];
      for (int i = 0; i < oBiases.Length; ++i)
        result[k++] = oBiases[i];
      return result;
    }


    public NeuralNetwork(int numInput, int numHidden, int numOutput)
    {
        this.numInput = numInput;
        this.numHidden = numHidden;
        this.numOutput = numOutput;

        inputs = new double[numInput];

        ihWeights = MakeMatrix(numInput, numHidden, 0.0); // input * hidden weights
        hBiases = new double[numHidden];
        hOutputs = new double[numHidden];

        oBiases = new double[numOutput];
        outputs = new double[numOutput];
        hoWeights = MakeMatrix(numHidden, numOutput, 0.0); // hidden * output weights

        rnd = new Random(1010101);
        initializeWeights();
    }

    public void SetWeights(double[] weights)
    {
        if (weights.Length != numWeights)
            throw new Exception("Bad weights array in NeuralNetwork constructor SetWeights array is");

        int k = 0; // weight param points


        for (int i = 0; i < numInput; i++)
        {
            for (int j = 0; j < numHidden; j++)
            {
                ihWeights[i][j] = weights[k++];
            }
        }

        for (int i = 0; i < numHidden; i++)
        {
            hBiases[i] = weights[k++];
        }

        for (int i = 0; i < numHidden; i++)
        {
            for (int j = 0; j < numOutput; j++)
            {
                hoWeights[i][j] = weights[k++];
            }
        }
        

        for (int i = 0; i < numOutput; i++)
        {
            oBiases[i] = weights[k++];
        }
    }  
    
    private static double[][] MakeMatrix(int rows, int cols, double v)
    {
        double[][] result = new double[rows][];
        for (int r = 0; r < result.Length; r++)
        {
            result[r] = new double[cols];
            for (int c = 0; c < result[r].Length; c++)
            {
                result[r][c] = v;
            }
        }
        return result;
    }

    private void initializeWeights()
    {
        // initialize to small random values
        double[] initialWeights = new double[numWeights];
        for (int i = 0; i < initialWeights.Length; i++)
        {
            initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
        }
        SetWeights(initialWeights);
    }

    public double[] ComputeOutputs(double[] xValues)
    {
     
        if (xValues.Length != numInput)
            throw new Exception("Bad xValues array in NeuralNetwork ComputeOutputs");

        // sums prior to activation
        double[] hSums = new double[numHidden]; 
        double[] oSums = new double[numOutput]; 

        for (int i = 0; i < xValues.Length; i++) 
        {
            inputs[i] = xValues[i];
        }

        for (int j = 0; j < numHidden; j++)
        {
            for (int i = 0; i < numInput; i++)
            {
                hSums[j] += inputs[i] * ihWeights[i][j];
            }

            hSums[j] += hBiases[j]; // add bias to hidden sums

            hOutputs[j] = HyperTan(hSums[j]); // apply activation function
        }

        // compute hidden output sum of weights * hOutputs
        for (int j = 0; j < numOutput; j++)
        {
            for (int i = 0; i < numHidden; i++)
            {
                oSums[j] += hOutputs[i] * hoWeights[i][j];
            }

            oSums[j] += oBiases[j]; // add biases to output sums
        }

        double[] softOut = Softmax(oSums); // optimize by combining all outputs
        Array.Copy(softOut, outputs, softOut.Length);

        double[] retResult = new double[numOutput];
        Array.Copy(outputs, retResult, retResult.Length);

        return retResult;
    }
    
    private static double HyperTan(double x)
    {
        if (x < -20.0) return -1.0;
        else if (x > 20.0) return 1.0;
        else return Math.Tanh(x);
    }

    private static double[] Softmax(double[] oSums)
    {
      // does all output nodes at once so scale
      // doesn't have to be re-computed each time

      double sum = 0.0;
      for (int i = 0; i < oSums.Length; ++i)
        sum += Math.Exp(oSums[i]);

      double[] result = new double[oSums.Length];
      for (int i = 0; i < oSums.Length; ++i)
        result[i] = Math.Exp(oSums[i]) / sum;

      return result; // now scaled so that xi sum to 1.0
    }

    // public double[] Train(double[][] trainData, int maxEpochs,
    //   double learnRate, double momentum)

    public double[] Train(double[][] trainData, int maxEpochs,
      double learnRate, double momentum, double weightDecay = 0.0001)
    {
      // train using back-prop
      // back-prop specific arrays
      double[][] hoGrads = MakeMatrix(numHidden, numOutput, 0.0); // hidden-to-output weight gradients
      double[] obGrads = new double[numOutput];                   // output bias gradients

      double[][] ihGrads = MakeMatrix(numInput, numHidden, 0.0);  // input-to-hidden weight gradients
      double[] hbGrads = new double[numHidden];                   // hidden bias gradients

      double[] oSignals = new double[numOutput];                  // local gradient output signals - gradients w/o associated input terms
      double[] hSignals = new double[numHidden];                  // local gradient hidden node signals

      // back-prop momentum specific arrays 
      double[][] ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
      double[] hPrevBiasesDelta = new double[numHidden];
      double[][] hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
      double[] oPrevBiasesDelta = new double[numOutput];

      int epoch = 0;
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // target values
      double derivative = 0.0;
      double errorSignal = 0.0;

      int[] sequence = new int[trainData.Length];
      for (int i = 0; i < sequence.Length; ++i)
        sequence[i] = i;

      int errInterval = maxEpochs / 10; // interval to check error
      while (epoch < maxEpochs)
      {
        ++epoch;

        if (epoch % errInterval == 0 && epoch < maxEpochs)
        {
          double trainErr = Error(trainData);
          Console.WriteLine("epoch = " + epoch + "  error = " +
            trainErr.ToString("F4"));
          //Console.ReadLine();
        }

        Shuffle(sequence); // visit each training data in random order
        for (int ii = 0; ii < trainData.Length; ++ii)
        {
          int idx = sequence[ii];
          Array.Copy(trainData[idx], xValues, numInput);
          Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
          ComputeOutputs(xValues); // copy xValues in, compute outputs 

          // indices: i = inputs, j = hiddens, k = outputs

          for (int k = 0; k < numOutput; ++k)
          {
            errorSignal = tValues[k] - outputs[k];  // Wikipedia uses (o-t)

            // Mean Squared Error with Softmax)
            // derivative = (1 - outputs[k]) * outputs[k]; // for softmax with MSE
            // oSignals[k] = errorSignal * derivative;

            // Cross-Entropy Error with Softmax
            // When cross-entropy error is paired with softmax activation,
            // the derivative term cancels out mathematically: oSignals[k] = errorSignal * 1.0
            oSignals[k] = errorSignal;
          }
  
          // compute hidden-to-output weight gradients using output signals
          for (int j = 0; j < numHidden; ++j)
            for (int k = 0; k < numOutput; ++k)
               hoGrads[j][k] = oSignals[k] * hOutputs[j];
 
          // compute output bias gradients using output signals
          for (int k = 0; k < numOutput; ++k)
            obGrads[k] = oSignals[k] * 1.0; // dummy assoc. input value

          // compute hidden node signals
          for (int j = 0; j < numHidden; ++j)
          {
            derivative = (1 + hOutputs[j]) * (1 - hOutputs[j]); // for tanh
            double sum = 0.0; // need sums of output signals times hidden-to-output weights
            for (int k = 0; k < numOutput; ++k) {
               sum += oSignals[k] * hoWeights[j][k]; // represents error signal
            }
            hSignals[j] = derivative * sum; 
          }

          // compute input-hidden weight gradients
          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
              ihGrads[i][j] = hSignals[j] * inputs[i];

          // compute hidden node bias gradients
          for (int j = 0; j < numHidden; ++j)
            hbGrads[j] = hSignals[j] * 1.0; // dummy 1.0 input

          // update weights and biases

          // update input-to-hidden weights
          for (int i = 0; i < numInput; ++i)
          {
            for (int j = 0; j < numHidden; ++j)
            {
              double delta = ihGrads[i][j] * learnRate;
              ihWeights[i][j] += delta; // would be -= if (o-t)
              ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;

              // Weight Decay for Input-to-Hidden Weights
              ihWeights[i][j] -= (weightDecay * ihWeights[i][j]);

              ihPrevWeightsDelta[i][j] = delta; // save for next time
            }
          }

          // update hidden biases
          for (int j = 0; j < numHidden; ++j)
          {
            double delta = hbGrads[j] * learnRate;
            hBiases[j] += delta;
            hBiases[j] += hPrevBiasesDelta[j] * momentum;
            hPrevBiasesDelta[j] = delta;
          }

          // update hidden-to-output weights
          for (int j = 0; j < numHidden; ++j)
          {
            for (int k = 0; k < numOutput; ++k)
            {
              double delta = hoGrads[j][k] * learnRate;
              hoWeights[j][k] += delta;
              hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;

              // Weight Decay for Hidden-to-Output Weights
              hoWeights[j][k] -= (weightDecay * hoWeights[j][k]);

              hoPrevWeightsDelta[j][k] = delta;
            }
          }

          // update output node biases
          for (int k = 0; k < numOutput; ++k)
          {
            double delta = obGrads[k] * learnRate;
            oBiases[k] += delta;
            oBiases[k] += oPrevBiasesDelta[k] * momentum;
            oPrevBiasesDelta[k] = delta;
          }

        } // each training item

      } // while
      double[] bestWts = GetWeights();
      return bestWts;
    } // Train

    private void Shuffle(int[] sequence) // instance method
    {
      for (int i = 0; i < sequence.Length; ++i)
      {
        int r = this.rnd.Next(i, sequence.Length);
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }
    } // Shuffle

    private double Error(double[][] trainData)
    {
      /*
      // average squared error per training item
      double sumSquaredError = 0.0;
      double[] xValues = new double[numInput]; // first numInput values in trainData
      double[] tValues = new double[numOutput]; // last numOutput values

      // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
      for (int i = 0; i < trainData.Length; ++i)
      {
        Array.Copy(trainData[i], xValues, numInput);
        Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
        double[] yValues = this.ComputeOutputs(xValues); // outputs using current weights
        for (int j = 0; j < numOutput; ++j)
        {
          double err = tValues[j] - yValues[j];
          sumSquaredError += err * err;
        }
      }
      return sumSquaredError / trainData.Length;
      */

      // average cross-entropy error per training item
      double sum = 0.0;
      double[] xValues = new double[numInput]; // first numInput values in trainData
      double[] tValues = new double[numOutput]; // last numOutput values

      for (int i = 0; i < trainData.Length; ++i)
      {
        Array.Copy(trainData[i], xValues, numInput);
        Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
        double[] yValues = this.ComputeOutputs(xValues); // outputs using current weights
        for (int j = 0; j < numOutput; ++j)
        {
          if (tValues[j] == 1.0) // 1-of-N target classification
          {
            sum += Math.Log(yValues[j]);
          }
        }
      }
      return -1.0 * sum / trainData.Length;
    } // Error

    public double Accuracy(double[][] testData)
    {
      // percentage correct using winner-takes all
      int numCorrect = 0;
      int numWrong = 0;
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // targets
      double[] yValues; // computed Y

      for (int i = 0; i < testData.Length; ++i)
      {
        Array.Copy(testData[i], xValues, numInput); // get x-values
        Array.Copy(testData[i], numInput, tValues, 0, numOutput); // get t-values
        yValues = this.ComputeOutputs(xValues);
        int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?
        int tMaxIndex = MaxIndex(tValues);

        if (maxIndex == tMaxIndex)
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / (numCorrect + numWrong);
    }

    private static int MaxIndex(double[] vector) // helper for Accuracy()
    {
      // index of largest value
      int bigIndex = 0;
      double biggestVal = vector[0];
      for (int i = 0; i < vector.Length; ++i)
      {
        if (vector[i] > biggestVal)
        {
          biggestVal = vector[i];
          bigIndex = i;
        }
      }
      return bigIndex;
    }
    
    
}