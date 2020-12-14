#ifndef __HELPER_HUNGARIAN__
#define __HELPER_HUNGARIAN__

///////////////////////////////////////////////////////////////////////////////
// Hungarian.cpp: Implementation file for Class HungarianAlgorithm.
// 
// This is a C++ wrapper with slight modification of a hungarian algorithm implementation by Markus Buehren.
// The original implementation is a few mex-functions for use in MATLAB, found here:
// http://www.mathworks.com/matlabcentral/fileexchange/6543-functions-for-the-rectangular-assignment-problem
// 
// Both this code and the orignal code are published under the BSD license.
// by Cong Ma, 2016
// 

#include <stdlib.h>
#include <cfloat> // for DBL_MAX
#include <cmath>  // for fabs()

namespace helper
{

  // template<typename D, typename IDX>
  //   D hungarian_solve(std::vector <std::vector<D> >& DistMatrix, std::vector<IDX>& Assignment);

  template<typename D, typename IDX>
  D hungarian_assignmentoptimal(IDX *assignment, D *distMatrix, IDX nOfRows, IDX nOfColumns);

  template<typename IDX>
  void hungarian_buildassignmentvector(IDX *assignment, bool *starMatrix, IDX nOfRows, IDX nOfColumns);

  template<typename D, typename IDX>
  void hungarian_computeassignmentcost(IDX *assignment, D *cost, D *distMatrix, IDX nOfRows);

  template<typename D, typename IDX>
  void hungarian_step2a(IDX *assignment, D *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, IDX nOfRows, IDX nOfColumns, IDX minDim);

  template<typename D, typename IDX>
  void hungarian_step2b(IDX *assignment, D *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, IDX nOfRows, IDX nOfColumns, IDX minDim);

  template<typename D, typename IDX>
  void hungarian_step3(IDX *assignment, D *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, IDX nOfRows, IDX nOfColumns, IDX minDim);

  template<typename D, typename IDX>
  void hungarian_step4(IDX *assignment, D *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, IDX nOfRows, IDX nOfColumns, IDX minDim, IDX row, IDX col);

  template<typename D, typename IDX>
  void hungarian_step5(IDX *assignment, D *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, IDX nOfRows, IDX nOfColumns, IDX minDim);

//********************************************************//
// A single function wrapper for solving assignment problem.
//********************************************************//
  template<typename D, typename IDX>
D hungarian_solve(std::vector <std::vector<D> >& DistMatrix, std::vector<IDX>& Assignment)
{
  IDX nRows = DistMatrix.size();
  IDX nCols = DistMatrix[0].size();

  D *distMatrixIn = new D[nRows * nCols];
  IDX *assignment = new IDX[nRows];
  D cost = 0.0;

  // Fill in the distMatrixIn. Mind the index is "i + nRows * j".
  // Here the cost matrix of size MxN is defined as a D precision array of N*M elements. 
  // In the solving functions matrices are seen to be saved MATLAB-IDXernally in row-order.
  // (i.e. the matrix [1 2; 3 4] will be stored as a vector [1 3 2 4], NOT [1 2 3 4]).
  for (IDX i = 0; i < nRows; i++)
    for (IDX j = 0; j < nCols; j++)
      distMatrixIn[i + nRows * j] = DistMatrix[i][j];
	
  // call solving function
  cost = hungarian_assignmentoptimal(assignment, distMatrixIn, nRows, nCols);

  Assignment.clear();
  for (IDX r = 0; r < nRows; r++)
    Assignment.push_back(assignment[r]);

  delete[] distMatrixIn;
  delete[] assignment;
  return cost;
}


//********************************************************//
// Solve optimal solution for assignment problem using Munkres algorithm, also known as Hungarian Algorithm.
//********************************************************//
  template<typename D, typename IDX>
D hungarian_assignmentoptimal(IDX *assignment, D *distMatrixIn, IDX nOfRows, IDX nOfColumns)
{
  D *distMatrix, *distMatrixTemp, *distMatrixEnd, *columnEnd, value, minValue;
  bool *coveredColumns, *coveredRows, *starMatrix, *newStarMatrix, *primeMatrix;
  IDX nOfElements, minDim, row, col;

  /* initialization */
  D cost = 0;
  for (row = 0; row<nOfRows; row++)
    assignment[row] = -1;

  /* generate working copy of distance Matrix */
  /* check if all matrix elements are positive */
  nOfElements = nOfRows * nOfColumns;
  distMatrix = (D *)malloc(nOfElements * sizeof(D));
  distMatrixEnd = distMatrix + nOfElements;

  for (row = 0; row<nOfElements; row++)
    {
      value = distMatrixIn[row];
      if (value < 0)
	std::cerr << "All matrix elements have to be non-negative." << std::endl;
      distMatrix[row] = value;
    }


  /* memory allocation */
  coveredColumns = (bool *)calloc(nOfColumns, sizeof(bool));
  coveredRows = (bool *)calloc(nOfRows, sizeof(bool));
  starMatrix = (bool *)calloc(nOfElements, sizeof(bool));
  primeMatrix = (bool *)calloc(nOfElements, sizeof(bool));
  newStarMatrix = (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4 */

  /* preliminary steps */
  if (nOfRows <= nOfColumns)
    {
      minDim = nOfRows;

      for (row = 0; row<nOfRows; row++)
	{
	  /* find the smallest element in the row */
	  distMatrixTemp = distMatrix + row;
	  minValue = *distMatrixTemp;
	  distMatrixTemp += nOfRows;
	  while (distMatrixTemp < distMatrixEnd)
	    {
	      value = *distMatrixTemp;
	      if (value < minValue)
		minValue = value;
	      distMatrixTemp += nOfRows;
	    }

	  /* subtract the smallest element from each element of the row */
	  distMatrixTemp = distMatrix + row;
	  while (distMatrixTemp < distMatrixEnd)
	    {
	      *distMatrixTemp -= minValue;
	      distMatrixTemp += nOfRows;
	    }
	}

      /* Steps 1 and 2a */
      for (row = 0; row<nOfRows; row++)
	for (col = 0; col<nOfColumns; col++)
	  if (fabs(distMatrix[row + nOfRows*col]) < DBL_EPSILON)
	    if (!coveredColumns[col])
	      {
		starMatrix[row + nOfRows*col] = true;
		coveredColumns[col] = true;
		break;
	      }
    }
  else /* if(nOfRows > nOfColumns) */
    {
      minDim = nOfColumns;

      for (col = 0; col<nOfColumns; col++)
	{
	  /* find the smallest element in the column */
	  distMatrixTemp = distMatrix + nOfRows*col;
	  columnEnd = distMatrixTemp + nOfRows;

	  minValue = *distMatrixTemp++;
	  while (distMatrixTemp < columnEnd)
	    {
	      value = *distMatrixTemp++;
	      if (value < minValue)
		minValue = value;
	    }

	  /* subtract the smallest element from each element of the column */
	  distMatrixTemp = distMatrix + nOfRows*col;
	  while (distMatrixTemp < columnEnd)
	    *distMatrixTemp++ -= minValue;
	}

      /* Steps 1 and 2a */
      for (col = 0; col<nOfColumns; col++)
	for (row = 0; row<nOfRows; row++)
	  if (fabs(distMatrix[row + nOfRows*col]) < DBL_EPSILON)
	    if (!coveredRows[row])
	      {
		starMatrix[row + nOfRows*col] = true;
		coveredColumns[col] = true;
		coveredRows[row] = true;
		break;
	      }
      for (row = 0; row<nOfRows; row++)
	coveredRows[row] = false;

    }

  /* move to step 2b */
  hungarian_step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

  /* compute cost and remove invalid assignments */
  hungarian_computeassignmentcost(assignment, &cost, distMatrixIn, nOfRows);

  /* free allocated memory */
  free(distMatrix);
  free(coveredColumns);
  free(coveredRows);
  free(starMatrix);
  free(primeMatrix);
  free(newStarMatrix);

  return cost;
}

/********************************************************/
template<typename IDX>
void hungarian_buildassignmentvector(IDX *assignment, bool *starMatrix, IDX nOfRows, IDX nOfColumns)
{
  IDX row, col;

  for (row = 0; row<nOfRows; row++)
    for (col = 0; col<nOfColumns; col++)
      if (starMatrix[row + nOfRows*col])
	{
#ifdef ONE_INDEXING
	  assignment[row] = col + 1; /* MATLAB-Indexing */
#else
	  assignment[row] = col;
#endif
	  break;
	}
}

/********************************************************/
template<typename D, typename IDX>
void hungarian_computeassignmentcost(IDX *assignment, D *cost, D *distMatrix, IDX nOfRows)
{
  IDX row, col;

  for (row = 0; row<nOfRows; row++)
    {      
      col = assignment[row];      
      if (col >= 0)
	*cost += distMatrix[row + nOfRows*col];
    }
}

/********************************************************/
  template<typename D, typename IDX>
void hungarian_step2a(IDX *assignment, D *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, IDX nOfRows, IDX nOfColumns, IDX minDim)
{
  bool *starMatrixTemp, *columnEnd;
  IDX col;

  /* cover every column containing a starred zero */
  for (col = 0; col<nOfColumns; col++)
    {
      starMatrixTemp = starMatrix + nOfRows*col;
      columnEnd = starMatrixTemp + nOfRows;
      while (starMatrixTemp < columnEnd){
	if (*starMatrixTemp++)
	  {
	    coveredColumns[col] = true;
	    break;
	  }
      }
    }

  /* move to step 3 */
  hungarian_step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
    template<typename D, typename IDX>
void hungarian_step2b(IDX *assignment, D *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, IDX nOfRows, IDX nOfColumns, IDX minDim)
{
  IDX col, nOfCoveredColumns;

  /* count covered columns */
  nOfCoveredColumns = 0;
  for (col = 0; col<nOfColumns; col++)
    if (coveredColumns[col])
      nOfCoveredColumns++;

  if (nOfCoveredColumns == minDim)
    {
      /* algorithm finished */
      hungarian_buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    }
  else
    {
      /* move to step 3 */
      hungarian_step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }

}

/********************************************************/
      template<typename D, typename IDX>
void hungarian_step3(IDX *assignment, D *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, IDX nOfRows, IDX nOfColumns, IDX minDim)
{
  bool zerosFound;
  IDX row, col, starCol;

  zerosFound = true;
  while (zerosFound)
    {
      zerosFound = false;
      for (col = 0; col<nOfColumns; col++)
	if (!coveredColumns[col])
	  for (row = 0; row<nOfRows; row++)
	    if ((!coveredRows[row]) && (fabs(distMatrix[row + nOfRows*col]) < DBL_EPSILON))
	      {
		/* prime zero */
		primeMatrix[row + nOfRows*col] = true;

		/* find starred zero in current row */
		for (starCol = 0; starCol<nOfColumns; starCol++)
		  if (starMatrix[row + nOfRows*starCol])
		    break;

		if (starCol == nOfColumns) /* no starred zero found */
		  {
		    /* move to step 4 */
		    hungarian_step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
		    return;
		  }
		else
		  {
		    coveredRows[row] = true;
		    coveredColumns[starCol] = false;
		    zerosFound = true;
		    break;
		  }
	      }
    }

  /* move to step 5 */
  hungarian_step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
	template<typename D, typename IDX>
void hungarian_step4(IDX *assignment, D *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, IDX nOfRows, IDX nOfColumns, IDX minDim, IDX row, IDX col)
{
  IDX n, starRow, starCol, primeRow, primeCol;
  IDX nOfElements = nOfRows*nOfColumns;

  /* generate temporary copy of starMatrix */
  for (n = 0; n<nOfElements; n++)
    newStarMatrix[n] = starMatrix[n];

  /* star current zero */
  newStarMatrix[row + nOfRows*col] = true;

  /* find starred zero in current column */
  starCol = col;
  for (starRow = 0; starRow<nOfRows; starRow++)
    if (starMatrix[starRow + nOfRows*starCol])
      break;

  while (starRow<nOfRows)
    {
      /* unstar the starred zero */
      newStarMatrix[starRow + nOfRows*starCol] = false;

      /* find primed zero in current row */
      primeRow = starRow;
      for (primeCol = 0; primeCol<nOfColumns; primeCol++)
	if (primeMatrix[primeRow + nOfRows*primeCol])
	  break;

      /* star the primed zero */
      newStarMatrix[primeRow + nOfRows*primeCol] = true;

      /* find starred zero in current column */
      starCol = primeCol;
      for (starRow = 0; starRow<nOfRows; starRow++)
	if (starMatrix[starRow + nOfRows*starCol])
	  break;
    }

  /* use temporary copy as new starMatrix */
  /* delete all primes, uncover all rows */
  for (n = 0; n<nOfElements; n++)
    {
      primeMatrix[n] = false;
      starMatrix[n] = newStarMatrix[n];
    }
  for (n = 0; n<nOfRows; n++)
    coveredRows[n] = false;

  /* move to step 2a */
  hungarian_step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
  template<typename D, typename IDX>
void hungarian_step5(IDX *assignment, D *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, IDX nOfRows, IDX nOfColumns, IDX minDim)
{
  D h, value;
  IDX row, col;

  /* find smallest uncovered element h */
  h = DBL_MAX;
  for (row = 0; row<nOfRows; row++)
    if (!coveredRows[row])
      for (col = 0; col<nOfColumns; col++)
	if (!coveredColumns[col])
	  {
	    value = distMatrix[row + nOfRows*col];
	    if (value < h)
	      h = value;
	  }

  /* add h to each covered row */
  for (row = 0; row<nOfRows; row++)
    if (coveredRows[row])
      for (col = 0; col<nOfColumns; col++)
	distMatrix[row + nOfRows*col] += h;

  /* subtract h from each uncovered column */
  for (col = 0; col<nOfColumns; col++)
    if (!coveredColumns[col])
      for (row = 0; row<nOfRows; row++)
	distMatrix[row + nOfRows*col] -= h;

  /* move to step 3 */
  hungarian_step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

  /*
    With this package, I provide some MATLAB-functions regarding the rectangular assignment problem. This problem appears for example in tracking applications, where one has M existing tracks and N new measurements. For each possible assignment, a cost or distance is computed. All cost values form a matrix, where the row index corresponds to the tracks and the column index corresponds to the measurements. The provided functions return an optimal or suboptimal assignment - in the sense of minimum overall costs - for the given matrix.
    In the process of gating, typically very unlikely assignments are forbidden. The given functions can handle forbidden assignments, which are marked by setting the corresponding assignment cost to infinity.
    The optimal solution is computed using Munkres algorithm, also known as Hungarian Algorithm.
   */

  /*
    0,0	1,2	2,3	3,4
    cost: 31
   */

    template<typename I>
    I hungarian_costMatrixIdx(const I source, const I target, const I nSources)
    {
      assert(source < nSources);
      return source+nSources*target;
    }
      
void hungarian_test()
{
  using IDX=int;
  using D=double;
  
  // please use "-std=c++11" for this initialization of vector.

  IDX nSources=4;
  IDX nTargets=5;
  // std::vector<D> costMatrix =
  //   {10, 19, 8, 15, 0 , 
  //    10, 18, 7, 17, 0 , 
  //    13, 16, 9, 14, 0 , 
  //    12, 19, 8, 18, 0
  //   };

  std::vector<D> costMatrix =
    {
      10, 10, 13, 12,
      19, 18, 16, 19,
      8, 7, 9, 8,
      15, 17, 14, 18,
      0, 0, 0, 0
    };

  for(IDX s=0; s<nSources; s++)
    for(IDX t=0; t<nTargets; t++)
      std::cout << "cost of assigning source " << s << " to target " << t << ": "
		<< costMatrix[hungarian_costMatrixIdx(s,t,nSources)]
		<< std::endl;;

  assert(costMatrix.size()==nSources*nTargets);
  
  std::vector<IDX> assignment(nSources, -1);

  //D cost = hungarian_solve(costMatrix, assignment);

  D cost=hungarian_assignmentoptimal
    (&assignment[0], &costMatrix[0], nSources, nTargets);

  for (unsigned int x = 0; x < nSources; x++)
    std::cout << x << "," << assignment[x] << "\t";

  std::cout << "\ncost: " << cost << std::endl;
}

}

#endif // __HELPER_HUNGARIAN__
