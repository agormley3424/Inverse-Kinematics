#include "IK.h"
#include "FK.h"
#include "minivectorTemplate.h"
#include <Eigen/Dense>
#include <adolc/adolc.h>
#include <cassert>
#include "mat3d.h"
#if defined(_WIN32) || defined(WIN32)
  #ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
  #endif
#endif
#include <math.h>
using namespace std;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

namespace
{
// Converts degrees to radians.
template<typename real>
inline real deg2rad(real deg) { return deg * M_PI / 180.0; }

template<typename real>
Mat3<real> Euler2Rotation(const real angle[3], RotateOrder order)
{
  Mat3<real> RX = Mat3<real>::getElementRotationMatrix(0, deg2rad(angle[0]));
  Mat3<real> RY = Mat3<real>::getElementRotationMatrix(1, deg2rad(angle[1]));
  Mat3<real> RZ = Mat3<real>::getElementRotationMatrix(2, deg2rad(angle[2]));

  switch(order)
  {
    case RotateOrder::XYZ:
      return RZ * RY * RX;
    case RotateOrder::YZX:
      return RX * RZ * RY;
    case RotateOrder::ZXY:
      return RY * RX * RZ;
    case RotateOrder::XZY:
      return RY * RZ * RX;
    case RotateOrder::YXZ:
      return RZ * RX * RY;
    case RotateOrder::ZYX:
      return RX * RY * RZ;
  }
  assert(0);
}

Mat3d* Mat3ToMat3d(Mat3<double> templateMat)
{
    double* convertedMatrix;
    templateMat.convertToArray(convertedMatrix);

    return new Mat3d(convertedMatrix);
}

template <typename real>
void PrintArrayAsMatrix(real* mat, string name)
{
    cout << name << ":\n";
    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 3; c++)
        {
            cout << mat[(r * 3) + c] << ", ";
        }
        cout << endl;
    }
}

template <typename real>
void PrintArrayAsVector(real* vec, string name)
{
    cout << name << ":\n";
    for (int r = 0; r < 3; r++)
    {
        cout << vec[r] << ", ";
    }
    cout << endl;
}

template <typename real>
void PrintMatrix(Mat3<real> mat, string name)
{
    cout << name << ":\n";
    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 3; c++)
        {
            cout << mat[r][c] << ", ";
        }
        cout << endl;
    }
}

void PrintMatrix(Mat3d mat, string name)
{
    cout << name << ":\n";
    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 3; c++)
        {
            cout << mat[r][c] << ", ";
        }
        cout << endl;
    }
}

void PrintMatrix(Eigen::MatrixXd mat, string name)
{
    cout << name << ":\n";
    for (int r = 0; r < mat.rows(); r++)
    {
        for (int c = 0; c < mat.cols(); c++)
        {
            cout << mat(r, c) << ' ';
        }

        cout << endl;
    }
}

void PrintVector(Eigen::VectorXd vec, string name)
{
    cout << name << ":\n";
    for (int r = 0; r < vec.rows(); r++)
    {
        cout << vec(r) << endl;
    }
}

void PrintVector(Vec3d vec, string name)
{
    cout << name << ":\n";
    for (int i = 0; i < 3; i++)
    {
        cout << vec[i] << ", ";
    }
    cout << endl;
}

template <typename real>
void PrintVector(Vec3<real> vec, string name)
{
    cout << name << ":\n";
    for (int i = 0; i < 3; i++)
    {
        cout << vec[i] << ", ";
    }
    cout << endl;
}

template <typename real>
Vec3<real> Vec3DToVec3(const Vec3d vec)
{
    Vec3<real> returnVec;

    returnVec[0] = vec[0];
    returnVec[1] = vec[1];
    returnVec[2] = vec[2];

    return returnVec;
}
// Performs forward kinematics, using the provided "fk" class.
// This is the function whose Jacobian matrix will be computed using adolc.
// numIKJoints and IKJointIDs specify which joints serve as handles for IK:
//   IKJointIDs is an array of integers of length "numIKJoints"
// Input: numIKJoints, IKJointIDs, fk, eulerAngles (of all joints)
// Output: handlePositions (world-coordinate positions of all the IK joints; length is 3 * numIKJoints)
template<typename real>
void forwardKinematicsFunction(
    int numIKJoints, const int * IKJointIDs, const FK & fk,
    const std::vector<real> & eulerAngles, std::vector<real> & handlePositions)
{
    // Students should implement this.
    // The implementation of this function is very similar to function computeLocalAndGlobalTransforms in the FK class.
    // The recommended approach is to first implement FK::computeLocalAndGlobalTransforms.
    // Then, implement the same algorithm into this function. To do so,
    // you can use fk.getJointUpdateOrder(), fk.getJointRestTranslation(), and fk.getJointRotateOrder() functions.
    // Also useful is the multiplyAffineTransform4ds function in minivectorTemplate.h .
    // It would be in principle possible to unify this "forwardKinematicsFunction" and FK::computeLocalAndGlobalTransforms(),
    // so that code is only written once. We considered this; but it is actually not easily doable.
    // If you find a good approach, feel free to document it in the README file, for extra credit.

    // I need to find the global transformation matrices for all handle joints at specified angles (eulerAngles)
    // Then transform the rest translation point of each handle with its corresponding joint matrix

    // Angles are relative to the bind pose relative to the joint orientation
    // When a multiply a point by the bind pose matrix it's now relative to the joint orientation
    // When I multiply by the joint orientation matrix it's now relative to the world

    ///// Compute global (local point to world) transforms for each joint /////

    int numJoints = fk.getNumJoints();

    //RigidTransform4d* localTransforms = new RigidTransform4d[numJoints];
    //RigidTransform4d* globalTransforms = new RigidTransform4d[numJoints];

    Vec3<real>* localTranslations = new Vec3<real>[numJoints];
    Mat3<real>* localRotations = new Mat3<real>[numJoints];
    Vec3<real>* globalTranslations = new Vec3<real>[numJoints];
    Mat3<real>* globalRotations = new Mat3<real>[numJoints];

    // Overall joints
    for (int i = 0; i < numJoints; i++)
    {
        int currentJointIndex = fk.getJointUpdateOrder(i);

        // Local rotations relative to 0 angles bind
        Mat3<real> anat_incorrect_localR = Euler2Rotation<real>(&eulerAngles.data()[i * 3], fk.getJointRotateOrder(currentJointIndex));

        // PrintMatrix(anat_incorrect_localR, "Anat incorrect matrix");

        // Joint orient mapping
        const double* eulerArray = fk.getJointOrient(currentJointIndex).data();
        real aEulerArray[3] = { eulerArray[0], eulerArray[1], eulerArray[2] };
        Mat3<real> joint_orient = Euler2Rotation(aEulerArray, fk.getJointRotateOrder(currentJointIndex));

        // PrintMatrix(joint_orient, "Joint orient");

        // Anatomically correct rotation
        Mat3<real> anat_correct_localR = joint_orient * anat_incorrect_localR;

        // PrintMatrix(anat_correct_localR, "Local Rotation Matrix");
        //PrintVector(fk.getJointRestTranslation(currentJointIndex), "Translation Vector");
        
        // Combine rotation with translation from rest
        //localTransforms[currentJointIndex] = RigidTransform4d(anat_correct_localR, fk.getJointRestTranslation(currentJointIndex));
        localTranslations[currentJointIndex] = Vec3DToVec3<real>(fk.getJointRestTranslation(currentJointIndex));
        localRotations[currentJointIndex] = anat_correct_localR;

        //PrintMatrix(localTransforms[currentJointIndex].getRotation(), "Local Rotation Matrix");
        //PrintVector(localTransforms[currentJointIndex].getTranslation(), "Local Translation Vector");

        // Calculate and store global transforms
        int parent_i = fk.getJointParent(currentJointIndex);
        if (parent_i >= 0)
        {
            // globalTransforms[currentJointIndex] = globalTransforms[parent_i] * localTransforms[currentJointIndex];
            globalRotations[currentJointIndex] = globalRotations[parent_i] * localRotations[currentJointIndex];
            globalTranslations[currentJointIndex] = (globalRotations[parent_i] * localTranslations[currentJointIndex]) + globalTranslations[parent_i];
        }
        else
        {
            // globalTransforms[currentJointIndex] = localTransforms[currentJointIndex];
            globalTranslations[currentJointIndex] = localTranslations[currentJointIndex];
            globalRotations[currentJointIndex] = localRotations[currentJointIndex];
        }
    }

    ///// Move every IK handle according to its global transform /////

    for (int i = 0; i < numIKJoints; i++)
    {
        int jointIndex = IKJointIDs[i];
        // Vec3d newPoint = globalTransforms[jointIndex].transformPoint(fk.getJointRestTranslation(jointIndex));
        Vec3<real> newPoint = globalTranslations[jointIndex];
        handlePositions[i * 3] = newPoint[0];
        handlePositions[(i * 3) + 1] = newPoint[1];
        handlePositions[(i * 3) + 2] = newPoint[2];

        // handlePositions[i * 3] = 1.0;

        // PrintArrayAsVector(&handlePositions[i], "Handle position");

        // int j = 1;
    }

    delete[] localRotations;
    delete[] localTranslations;
    delete[] globalTranslations;
    delete[] globalRotations;
}

//template<typename real>
//void forwardKinematicsFunction(
//    int numIKJoints, const int* IKJointIDs, const FK& fk,
//    const std::vector<real>& eulerAngles, std::vector<real>& handlePositions)
//{
//    // Students should implement this.
//    // The implementation of this function is very similar to function computeLocalAndGlobalTransforms in the FK class.
//    // The recommended approach is to first implement FK::computeLocalAndGlobalTransforms.
//    // Then, implement the same algorithm into this function. To do so,
//    // you can use fk.getJointUpdateOrder(), fk.getJointRestTranslation(), and fk.getJointRotateOrder() functions.
//    // Also useful is the multiplyAffineTransform4ds function in minivectorTemplate.h .
//    // It would be in principle possible to unify this "forwardKinematicsFunction" and FK::computeLocalAndGlobalTransforms(),
//    // so that code is only written once. We considered this; but it is actually not easily doable.
//    // If you find a good approach, feel free to document it in the README file, for extra credit.
//
//    int n = fk.getNumJoints(); // total number of joints // to save computation everytime
//    // using vectors of Mat3<real> and Vec3<real> for passing as values to multiplyAffineTransform4ds
//    std::vector<Mat3<real>> localTransforms(n), globalTransforms(n); // for the transformations
//    std::vector<Vec3<real>> localTranslations(n), globalTranslations(n); // for the translations
//
//    // to compute localTransforms and localTranslations
//    for (int i = 0; i < n; i++)
//    {
//        Mat3<real> eulerRotMat, jointOEulerMat; // to store intermediate matrices
//        real angle[3];
//        // eulerAngles
//        angle[0] = eulerAngles[i * 3];
//        angle[1] = eulerAngles[i * 3 + 1];
//        angle[2] = eulerAngles[i * 3 + 2];
//        eulerRotMat = Euler2Rotation(angle, fk.getJointRotateOrder(i));
//        // jointOrientationEulerAngles
//        Vec3d jOrient = fk.getJointOrient(i);
//        angle[0] = jOrient.data()[0];
//        angle[1] = jOrient.data()[1];
//        angle[2] = jOrient.data()[2];
//        jointOEulerMat = Euler2Rotation(angle, XYZ);
//        // localTransform
//        localTransforms[i] = jointOEulerMat * eulerRotMat;
//        // translations
//        globalTranslations[i][0] = localTranslations[i][0] = fk.getJointRestTranslation(i)[0]; // the global translations here are temp, they will be calculated later
//        globalTranslations[i][1] = localTranslations[i][1] = fk.getJointRestTranslation(i)[1];
//        globalTranslations[i][2] = localTranslations[i][2] = fk.getJointRestTranslation(i)[2];
//    }
//
//    // to compute globalTransforms and globalTranslations
//    for (int i = 0; i < n; i++)
//    {
//        int current = fk.getJointUpdateOrder(i); // Get the joint that appears at position "i" in a linear joint update order.
//        int parentOfCurrent = fk.getJointParent(current);
//
//        if (parentOfCurrent == -1) // means current is root
//            globalTransforms[current] = localTransforms[current];
//        else
//            multiplyAffineTransform4ds(globalTransforms[parentOfCurrent], globalTranslations[parentOfCurrent], localTransforms[current], localTranslations[current], globalTransforms[current], globalTranslations[current]);
//    }
//
//    // to compute the handle positions
//    // std::cout << n <<' '<< numIKJoints <<"\n";
//    for (int i = 0; i < numIKJoints; i++) // numIKJoints is number of IK handles
//    { // The position of a handle is the global translation of the joint.
//        int jID = IKJointIDs[i];
//        handlePositions[i * 3] = globalTranslations[jID][0];
//        handlePositions[i * 3 + 1] = globalTranslations[jID][1];
//        handlePositions[i * 3 + 2] = globalTranslations[jID][2];
//    }
//}

} // end anonymous namespaces

IK::IK(int numIKJoints, const int * IKJointIDs, FK * inputFK, int adolc_tagID)
{
  this->numIKJoints = numIKJoints;
  this->IKJointIDs = IKJointIDs;
  this->fk = inputFK;
  this->adolc_tagID = adolc_tagID;

  FKInputDim = fk->getNumJoints() * 3;
  FKOutputDim = numIKJoints * 3;

  train_adolc();
}

void IK::train_adolc()
{
    // Students should implement this.
    // Here, you should setup adol_c:
    //   Define adol_c inputs and outputs. 
    //   Use the "forwardKinematicsFunction" as the function that will be computed by adol_c.
    //   This will later make it possible for you to compute the gradient of this function in IK::doIK
    //   (in other words, compute the "Jacobian matrix" J).
    // See ADOLCExample.cpp .

    // Begin recording
    trace_on(adolc_tagID);

    // Define and record inputs
    /* Could pass in the actual euler angles */
    vector<adouble> inputs(FKInputDim);
    for (int i = 0; i < FKInputDim; i++)
    {
        inputs[i] <<= 0.0;
    }

    // Define function output
    vector<adouble> func_out(FKOutputDim);

    // Define and record function
    /* Autodiv is an example of how this works for computational functions */
    /* Forms a graph of the function with each node as a basic mathematical operation, derivative is computed for each, then all are combined */
    forwardKinematicsFunction(numIKJoints, IKJointIDs, *fk, inputs, func_out);

    // Record output
    vector<double> outputs(FKOutputDim);
    for (int i = 0; i < FKOutputDim; i++)
    {
        func_out[i] >>= outputs[i];
    }

    // Stop recording
    trace_off();
}

Eigen::MatrixXd ArrToMatrix(double* array[], int rows, int cols)
{
    Eigen::MatrixXd returnMatrix(rows, cols);
    
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            returnMatrix(r, c) = array[r][c];
        }
    }

    return returnMatrix;
    
    //return RowMajorMatrix::Map(&array[0][0], rows, cols);
}

Eigen::VectorXd VecArrToVector(const Vec3d* array, int cols, int rows)
{
    //double* lin_targets = new double[cols * rows];

    //for (int i = 0; i < rows; i++)
    //{
    //    array[i].convertToArray(&lin_targets[i * 3]);
    //}

    //double* arr_targets[] = { lin_targets };

    //return Eigen::Map<Eigen::VectorXd>(&arr_targets[0][0], rows * cols);

    // return Eigen::Map<Eigen::VectorXd>(array->data(), rows, cols);

    return Eigen::Map<const Eigen::VectorXd>(&array[0][0], cols * rows);
}

//unsigned int vecSize = cols * rows;
//
//Eigen::VectorXd vec(vecSize);
//
//for (int i = 0; i < vecSize; i++)
//{
//    vec(i)
//}

void IK::doIK(const Vec3d * targetHandlePositions, Vec3d * jointEulerAngles)
{
    // Students should implement this.
    // Use adolc to evalute the forwardKinematicsFunction and its gradient (Jacobian). It was trained in train_adolc().
    // Specifically, use ::function, and ::jacobian .
    // See ADOLCExample.cpp .
    //
    // Use it implement the Tikhonov IK method (or the pseudoinverse method for extra credit).
    // Note that at entry, "jointEulerAngles" contains the input Euler angles. 
    // Upon exit, jointEulerAngles should contain the new Euler angles.

    int numJoints = fk->getNumJoints();

    // Convert target handle positions to eigen format
    // Eigen::VectorXd targetHandlePos = VecArrToVector(targetHandlePositions, numIKJoints, 3);
    Eigen::VectorXd targetHandlePos = VecArrToVector(targetHandlePositions, numIKJoints, 3);

    // Convert input euler angles to eigen format
    // Eigen::MatrixXd angles = VecArrToMatrix(jointEulerAngles, numJoints, 3);
    Eigen::VectorXd angles = VecArrToVector(jointEulerAngles, numJoints, 3);

    // Calculate current IK handle positions (in the current frame)
    Eigen::VectorXd currentHandlePositions(FKOutputDim);
    ::function(adolc_tagID, FKOutputDim, FKInputDim, angles.data(), currentHandlePositions.data());

    // Calculate jacobians
    double** jacobian_array = new double* [FKOutputDim];
    for (int i = 0; i < FKOutputDim; i++)
    {
        jacobian_array[i] = new double[FKInputDim];
    }

    /* I could use the .data field of a 2D eigen matrix for the output of this */
    ::jacobian(adolc_tagID, FKOutputDim, FKInputDim, angles.data(), jacobian_array);

    Eigen::MatrixXd jacobian = ArrToMatrix(jacobian_array, FKOutputDim, FKInputDim);

    // Eigen::MatrixXd jacobian = Eigen::Map<RowMajorMatrix>(*jacobian_array, FKOutputDim, FKInputDim);

    // Eigen::MatrixXd jacobian_T = Eigen::Map<Eigen::MatrixXd>(*jacobian_array, FKInputDim, FKOutputDim);

    Eigen::MatrixXd jacobian_T = jacobian.transpose();
    Eigen::MatrixXd j_product = jacobian_T * jacobian;

    // Calculate left matrix

    double alpha = 0.001;

    Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(j_product.rows(), j_product.cols());
    Eigen::MatrixXd left_matrix = j_product + (alpha * identity);

    // PrintMatrix(left_matrix, "Left Matrix");

    // Calculate right matrix

    Eigen::VectorXd handle_displacement = targetHandlePos - currentHandlePositions;
    Eigen::VectorXd right_matrix = jacobian_T * handle_displacement;

    //PrintVector(handle_displacement, "Handle displacement");

    //PrintMatrix(right_matrix, "Right Matrix");

    // Calculate new joints

    Eigen::VectorXd angle_displacement = left_matrix.ldlt().solve(right_matrix);
    Eigen::VectorXd new_angles = angles + angle_displacement;

    //PrintVector(angles, "Original angles");
    //PrintVector(angle_displacement, "Angle displacement");
    //PrintVector(new_angles, "New angles");

    for (int i = 0; i < numJoints; i++)
    {
        jointEulerAngles[i] = Vec3d(&new_angles.data()[i * 3]);
    }

    // Deallocate heap memory
    for (int i = 0; i < FKOutputDim; i++)
    {
        delete[] jacobian_array[i];
    }
    delete[] jacobian_array;
}
