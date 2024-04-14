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

    RigidTransform4d* localTransforms = new RigidTransform4d[numJoints];
    RigidTransform4d* globalTransforms = new RigidTransform4d[numJoints];

    // Overall joints
    for (int i = 0; i < numJoints; i++)
    {
        // Local rotations relative to 0 angles bind
        Mat3<real> anat_incorrect_localR = Euler2Rotation<adouble>(&eulerAngles.data()[i * 3], fk.getJointRotateOrder(i));

        // Joint orient mapping 
        const double* eulerArray = fk.getJointRestEulerAngles(i).data();
        real aEulerArray[3] = { eulerArray[0], eulerArray[1], eulerArray[2] };
        Mat3<real> joint_orient = Euler2Rotation(aEulerArray, fk.getJointRotateOrder(i));

        // Anatomically correct rotation
        Mat3<real> anat_correct_localR = anat_incorrect_localR * joint_orient;
        
        // Combine rotation with translation from rest
        localTransforms[i] = RigidTransform4d(anat_correct_localR, fk.getJointRestTranslation(i));

        // Calculate and store global transforms
        int parent_i = fk.getJointParent(i);
        if (parent_i >= 0)
        {
            globalTransforms[i] = globalTransforms[parent_i] * localTransforms[i];
        }
        else
        {
            globalTransforms[i] = localTransforms[i];
        }
    }

    ///// Move every IK handle according to its global transform /////

    for (int i = 0; i < numIKJoints; i++)
    {
        int jointIndex = IKJointIDs[i];
        Vec3d newPoint = globalTransforms[jointIndex].transformPoint(fk.getJointRestTranslation(jointIndex));
        handlePositions[i * 3] = newPoint[0];
        handlePositions[(i * 3) + 1] = newPoint[1];
        handlePositions[(i * 3) + 2] = newPoint[2];
    }

    delete[] localTransforms;
    delete[] globalTransforms;
}

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
    vector<adouble> inputs(FKInputDim);
    for (int i = 0; i < FKInputDim; i++)
    {
        inputs[i] <<= 0.0;
    }

    // Define function output
    vector<adouble> func_out = vector<adouble>(FKOutputDim);

    // Define and record function
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

Eigen::MatrixXd VecArrToMatrix(const Vec3d* array, int rows, int cols)
{
    double* lin_targets = new double[cols * rows];

    for (int i = 0; i < rows; i++)
    {
        array[i].convertToArray(&lin_targets[i * 3]);
    }

    double* arr_targets[] = { lin_targets };

    return Eigen::Map<Eigen::MatrixXd>(&arr_targets[0][0], rows, cols);
}

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

    //Vec3d testArr[2];
    //testArr[0] = { 0.2, 2.3, 1.4 };
    //testArr[1] = { 3.6, 8.3, 7.2 };
    //Eigen::MatrixXd testValue = VecArrToMatrix(testArr, 2, 3);

    //int numJoints = fk->getNumJoints();

    // Convert target handle positions to eigen format
    Eigen::VectorXd targetHandlePos = VecArrToMatrix(targetHandlePositions, numIKJoints * 3, 1);

    //// Convert input euler angles to eigen format
    //Eigen::MatrixXd angles = VecArrToMatrix(jointEulerAngles, numJoints, 3);

    //// Create vector of zeros for last positions
    //Eigen::VectorXd lastHandlePositions = Eigen::MatrixXd::Zero(numIKJoints * 3, 1);

    //// Calculate jacobians
    //double** jacobian_array = new double* [FKOutputDim];
    //for (int i = 0; i < FKOutputDim; i++)
    //{
    //    jacobian_array[i] = new double[FKInputDim];
    //}

    //::jacobian(adolc_tagID, FKOutputDim, FKInputDim, angles.data(), jacobian_array);

    //Eigen::MatrixXd jacobian = Eigen::Map<Eigen::MatrixXd>(*jacobian_array, FKOutputDim, FKInputDim);

    //Eigen::MatrixXd jacobian_T = jacobian.transpose();
    //Eigen::MatrixXd j_product = jacobian_T * jacobian;

    //// Calculate left matrix

    //double alpha = 0.001;

    //Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(j_product.rows(), j_product.cols());
    //Eigen::MatrixXd left_matrix = j_product + (alpha * identity);

    //// Calculate right matrix

    //Eigen::VectorXd handle_displacement = targetHandlePos - lastHandlePositions;
    //Eigen::MatrixXd right_matrix = jacobian_T * handle_displacement;

    //// Calculate new joints

    //Eigen::VectorXd angle_displacement = left_matrix.colPivHouseholderQr().solve(right_matrix);
    //Eigen::MatrixXd lin_angles = VecArrToMatrix(jointEulerAngles, numJoints * 3, 1);
    //Eigen::VectorXd new_angles = lin_angles + angle_displacement;

    //for (int i = 0; i < numJoints; i++)
    //{
    //    jointEulerAngles[i] = Vec3d(new_angles.data()[i * 3]);
    //}

    //delete[] jacobian_array;
}

