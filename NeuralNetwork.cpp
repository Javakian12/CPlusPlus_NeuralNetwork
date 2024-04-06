#include <iostream>
#include <cmath>
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <sstream>
#include <omp.h>
#include <random>

using namespace std;


// The architecture this program uses is a simple 2 input node network with an input layer, hidden layer, and output layer. The hidden and ouput layer has its respective bias nodes as well.

//learning rate
double r;
double t1;
double t2;

double yIn1;
double yIn2;

double sigmoid(double x) { //activation function
    return 1 / (1 + exp(-x));
}

double forwardProp(double x1, double x2, double weight1, double weight2, double biasNode, double newNode) //forward propigation from input to hidden layer
{
    //foward pass
    newNode = sigmoid((x1 * weight1) + (x2 * weight2) + biasNode);
    return newNode;
}

double forwardPropOutput(double x1, double x2, double weight1, double weight2, double newNode) //forward propigation from hidden to output layer
{
    newNode = (x1 * weight1) + (x2 * weight2);
    return newNode;
}

double error(double y1, double y2) //calculate error-loss
{
    return 0.5 * (pow((t1 - y1), 2) + pow((t2 - y2), 2));
}

double getYInput1(double h1, double h2, double w1, double w2)
{
    return (h1 * w1) + (h2 * w2); //reverse forward propigation
}

double backPropFirst(double w1, double w2, double * bias, double y2, double prevWeight, double* w_output, int output_size, double* h, int node, int mySpot, double y1) //back-propigation on the output layer to hidden
{
    double step1, step2;
    if (node % 2 == 0)
    {
        step2 = yIn1 * (1 - yIn1); //step2 of backpropigation
        step1 = -(t1 - y1); //step1 of backpropigation
    }
    else if (node % 2 > 0)
    {
        step2 = yIn2 * (1 - yIn2); //step2 of backpropigation
        step1 = -(t2 - y2); //step1 of backpropigation
    }
    //step3 is just correctH
    //backwards pass now
    double backPropWeight = step1 * step2 * h[mySpot]; //get weight

    return prevWeight - (r * backPropWeight); //adjust via learning rate and previous weight
    
}

double backPropHidden(double x1, double x2, double w, double* bias, double y1, double y2, double prevWeight1, double prevWeight2, double hInFinal, int node, int output_size, double* w_hidden, double x, int hidden1, int hidden2) //back-propigation on the hidden layer to input
{
    double step2 = 0.0;
    //step 1

    double first_step1 = -1 * (t1 - y1); //chain rule
    double first_step2 = yIn1 * (1 - yIn1); //second step
    double first_step3 = prevWeight1; //previous weight going into node
    double second_step1 = -1 * (t2 - y2); //chain rule
    double second_step2 = yIn2 * (1 - yIn2); //second step
    double second_step3 = prevWeight2; //previous weight going into node
    double final_first_step = (first_step1 * first_step2 * first_step3) + (second_step1 * second_step2 * second_step3); //bring them together
    //step 2 (hard)
  
    step2 += hInFinal * (1 - hInFinal);
    step2 *= x; //multiply propigation by node value

    //step 3
    double final_step = final_first_step * step2;
    
    return w - (r * final_step);
}


int main()
{
    double x1;
    double x2;
    //starting values
    double y1 = 0.0; //output layer node 1
    double y2 = 0.0; //output layer node 2

    //starting weights
    double w1 = 0.13;
    double w2 = 0.3;
    double w3 = 0.23;
    double w4 = 0.2;
    double w5 = 0.2;
    double w6 = 0.41;
    double w7 = 0.4;
    double w8 = 0.53;

    int thread_count;
    double error_loss;
    double* h;
    double* bias;
    int hidden_layer_size;
    double* w_hidden;
    double* w_output;

    int weightSize;
    int epochs;

        x1 = 0.02; //input node
        x2 = 0.23; //input node
        thread_count = 1; //numthreads
        h = new double[4]; //set hidden layer nodes
        bias = new double[6]; //set bias nodes (hidden layer size + 2)
        w_hidden = new double[8]; //set hidden layer weights (hidden layer size * 2)
        w_output = new double[8]; //set output layer weights (hidden layer size * 2)
        hidden_layer_size = 4; //hidden layer size (must be the same as the amound of hidden layer nodes)
        weightSize = 8; //amount of nodes in hidden layer
        epochs = 50; //epochs
        t1 = 0.02; //expected value
        t2 = 0.98; //expected value 
        r = 0.001; //learning rate

    int num; //random number between 25 and 45
    double dec; //random number between 0.25 and 0.45
    srand(time(NULL));
    for (int i = 0; i < hidden_layer_size + 2; i++)
    {
        num = rand() % 80; //get random number between 25 and 45
        dec = num / 100.0; //convert integer to decimal
        bias[i] = dec; //set bias
    }
    for (int i = 0; i < weightSize; i++)
    {
        num = rand() % 80 + 5; //get random number between 10 and 60
        dec = num / 100.0; //convert integer to decimal
        w_hidden[i] = dec; //set weight

        num = rand() % 80 + 15; //get random number between 10 and 60
        dec = num / 100.0; //convert integer to decimal
        w_output[i] = dec; //set weight
    }

    for (int i = 0; i < weightSize; i++) {
        cout << "Bias: " << bias[i] << endl;
        cout << "Hidden Weights: " << w_hidden[i] << endl;
        cout << "Output Weights: " << w_output[i] << endl;
    }
     
    
    //static values

    for (int i = 0; i < hidden_layer_size + 2; i++)
    {
        h[i] = 0.0; //initialize hidden layer nodes to 0.0 for the time being
    }

    cout << "Starting values are:\nx1: " << x1 << "\nx2: " << x2 << "\nExpected:\nt1: " << t1 << "\nt2: " << t2 << endl;
    double start;
    start = omp_get_wtime();//time checking
#pragma omp parallel num_threads(thread_count)
    {
        //srand(time(NULL)); //generate seed for random number generator
        
        int thread_id = omp_get_thread_num();
        double newNode;
        int doubleSkip = 0;
        int mySpot = thread_id; //for balancing threads in nodes
        for (int i = 0; i < epochs; i++)
        {
            mySpot = thread_id;
            //forward propigate the first node
            //thread operation
            for (int j = 0; j < hidden_layer_size; j++)
            {
                //check thread id
                //split up work by num_threads (thread_id 0 does node 0, thread_id 1 does node 1, and so on)
                if (mySpot < hidden_layer_size) //check if thread has a node to work on
                {
                    h[mySpot] = forwardProp(x1, x2, w_hidden[((mySpot + 1) * 2) - 2], w_hidden[((mySpot + 1) * 2)-1], bias[j], newNode);
                }
                mySpot += thread_count; //balance threads per nodes

            }
            //reset mySpot
            mySpot = thread_id;
            //wait for all threads to finish
#pragma omp barrier
            double newNode1;
            //forward propigation from hidden to output layer
            y1 = 0.0; //reset output values so new forward propigation can take place
            y2 = 0.0; //reset output values so new forward propigation can take place
            for (int f = 0; f < hidden_layer_size; f += 2) //f += 2 as we are consuming 2 weights per operation per thread
            {
                if (thread_count > 1)
                {
                    if (mySpot == 0) //first node (thread 0)
                    {
                        y1 += forwardPropOutput(h[f], h[f + 1], w_output[f], w_output[f + 1], newNode1); //first node, goes up to the middle of the weight list, but end of the node list
                    }
                    else if (mySpot == 1) //second node (thread 1)
                    {
                        y2 += forwardPropOutput(h[f], h[(f + 1)], w_output[(f + (weightSize / 2))], w_output[(f + (weightSize / 2) + 1)], newNode1);
                        
                    }
                }
                else if(thread_count == 1)
                {
                    y1 += forwardPropOutput(h[f], h[f + 1], w_output[f], w_output[f + 1], newNode1); //first node, goes up to the middle of the weight list, but end of the node list
                    y2 += forwardPropOutput(h[f], h[(f + 1)], w_output[(f + (weightSize / 2))], w_output[(f + (weightSize / 2) + 1)], newNode1);
                }
            }
            if (thread_count > 1) //if program is multithreaded
            {
                if (thread_id == 0) //first node (thread 0)
                {
                    y1 += bias[weightSize - 2];
                    y1 = sigmoid(y1);
                }
                else if (thread_id == 1) //second node (thread 1)
                {
                    y2 += bias[weightSize-1];
                    y2 = sigmoid(y2);
                }
            }
            else if (thread_count == 1) //if program is single threaded
            {
                y1 += bias[weightSize - 2];
                y1 = sigmoid(y1);
                y2 += bias[weightSize-1];
                y2 = sigmoid(y2);
            }

            //forward propigation done
            //calculate error
            stringstream str_epoch_run;
            if (thread_id < 1) //first node (thread 0)
            {
                error_loss = error(y1, y2); //calculate error of the forward propigation
                str_epoch_run << "Epoch " << i << " resulted in an error of: " << error_loss << endl;
                cout << str_epoch_run.str();
            }
            //others wait
#pragma omp barrier
            //begin back propigation
            double zero1 = 0.0;
            double zero2 = 0.0;
            yIn1 = zero1;
            yIn2 = zero2;
            //reset mySpot
            mySpot = thread_id;
            int weightSpot1 = 0;
            int weightSpot2 = 1;
            //get yIn1 and yIn2
            int weightNum = 0;
            for (int p = 0; p < hidden_layer_size/2; p++) //calculate reverse forward propigation
            {
                if (thread_count > 1)
                {
                    if (thread_id == 0) //thread 0
                    {
                        yIn1 += getYInput1(h[weightNum], h[weightNum + 1], w_output[weightNum], w_output[weightNum + 1]); //get reverse forward propigation without activation function (backpropigation step)
                    }
                    else if (thread_id == 1) //thread 1
                    {
                        yIn2 += getYInput1(h[weightNum], h[weightNum + 1], w_output[weightNum+hidden_layer_size], w_output[weightNum+ hidden_layer_size + 1]); //get reverse forward propigation without activation function (backpropigation step)
                    }
                }
                else //program is single threaded
                {
                    yIn1 += getYInput1(h[weightNum], h[weightNum + 1], w_output[weightNum], w_output[weightNum + 1]);

                    yIn2 += getYInput1(h[weightNum], h[weightNum + 1], w_output[weightNum+ hidden_layer_size], w_output[weightNum + hidden_layer_size + 1]);

                }
                weightNum += 2; //parallization technique
            }
            if (thread_count > 1) //if multithreaded
            {
                if (thread_id == 0)
                {
                    yIn1 += bias[hidden_layer_size]; //add bias 
                }
                else if (thread_id == 1)
                {
                    yIn2 += bias[hidden_layer_size + 1]; //add bias
                }
            }
            else //single threaded
            { 
                //add bias
                    yIn1 += bias[hidden_layer_size];
                    yIn2 += bias[hidden_layer_size + 1];
            }

            int nodeSpot = 0;
            weightNum = 0;
            mySpot = thread_id;
            for (int j = 0; j < (hidden_layer_size * 2); j++) //back propigation from output to hidden layer
            {
                if (nodeSpot > hidden_layer_size-1)
                {
                    nodeSpot = 0; //which input node we are back propigating to
                }
                if (mySpot < (hidden_layer_size * 2))
                {
                    if (mySpot % 2 == 0) //if even node (for getting correct weights)
                    {
                        w_output[mySpot] = backPropFirst(w_output[mySpot], w_output[mySpot + 1], bias, y2, w_output[mySpot], w_output, hidden_layer_size, h, mySpot, nodeSpot, y1);
                    }
                    else if (mySpot % 2 > 0) //if odd node (for getting correct weights)
                    {
                        w_output[mySpot] = backPropFirst(w_output[mySpot], w_output[mySpot - 1], bias, y2, w_output[mySpot], w_output, hidden_layer_size, h, mySpot, nodeSpot, y1);
                    }
                }
                mySpot += thread_count; //So we can correctly task each thread with a weight

                nodeSpot += 1;
                weightNum += 2;
            }
                
            //threads wait until all finishes (zig-zag issue in neural network otherwise)
#pragma omp barrier
            int myWeight = thread_id;
            mySpot = thread_id;
            double currNode = 0.0;
            double weightNewNode;
            mySpot = thread_id; //reset mySpot
            double hInPrime;

            for (int j = 0; j < (hidden_layer_size*2); j++) //second stage back propigation (hidden layer to input) (Per weight!!!)
            { 
                hInPrime = forwardProp(x1, x2, w_hidden[((mySpot + 1) * 2) - 2], w_hidden[((mySpot + 1) * 2)-1], bias[j], weightNewNode);
                
                if (mySpot % 2 == 0)
                {
                    currNode = x1; //we are backpropigating to input node 1
                }
                else
                {
                    currNode = x2; //we are backpropigating to input node 2
                }
                if (mySpot < (hidden_layer_size * 2)) 
                {
                    //back propigate from hidden layer to input
                    w_hidden[mySpot] = backPropHidden(x1, x2, w_hidden[mySpot], bias, y1, y2, w_output[((mySpot + 1) * 2) - 2], w_output[((mySpot + 1) * 2)-1], hInPrime, mySpot, hidden_layer_size, w_hidden, currNode, w_hidden[((mySpot + 1) * 2) - 2], w_hidden[((mySpot + 1) * 2)-1]);
                }
                mySpot += thread_count; //get next index for thread (the next weight to work on)
            }

#pragma omp barrier //restart forward propigation
            stringstream sa;
            sa << "\nThread ID: " << thread_id << " Reports the current weights: " << endl << endl;
            sa << "------- Weight Hidden layer 0: " << w_hidden[0] << ", 1 : " << w_hidden[1] << ", 2 : " << w_hidden[2] << ", 3 : " << w_hidden[3] << endl << endl;
            sa << "------- Weight Output layer 0: " << w_output[0] << ", 1: " << w_output[1] << ", 2: " << w_output[2] << ", 3: " << w_output[3] << endl << endl;
            sa << "------- Output y1: " << y1 << ", y2: " << y2 << ", expected y1 to be: " << t1 << ", expected y2 to be: " << t2 << endl << endl;
            cout << sa.str();

}
    }
    double end = omp_get_wtime();

    //display time
    cout << "Time Taken: " << end - start << " sec" << endl;
}
