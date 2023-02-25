/*
 * basis_q1.hpp
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

#ifndef DIFFUSION_PROBLEM_INCLUDE_Q1BASIS_HPP_
#define DIFFUSION_PROBLEM_INCLUDE_Q1BASIS_HPP_



// Deal.ii
#include <deal.II/base/function.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients
{
using namespace dealii;

template <int dim>
  class Q1Basis : public Function<dim>
  {
  public:
    Q1Basis() = delete;


    Q1Basis(const typename Triangulation<dim>::active_cell_iterator &cell);

    Q1Basis(const Q1Basis<dim> &);

    void
    set_index(unsigned int index);


    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;


    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<double> &          values,
               const unsigned int             component = 0) const override;

  private:

    unsigned int index_basis;

    FullMatrix<double> coeff_matrix;
  };

template <int dim>
 Q1Basis<dim>::Q1Basis(const Q1Basis<dim> &basis)
   : Function<dim>()
   , index_basis(0)
   , coeff_matrix(basis.coeff_matrix)
 {}


 template <>
 Q1Basis<2>::Q1Basis(
   const typename Triangulation<2>::active_cell_iterator &cell)
   : Function<2>()
   , index_basis(0)
   , coeff_matrix(4, 4)
 {
   FullMatrix<double> point_matrix(4, 4);

   for (unsigned int i = 0; i < 4; ++i)
     {
       const Point<2> &p = cell->vertex(i);

       point_matrix(i, 0) = 1;
       point_matrix(i, 1) = p(0);
       point_matrix(i, 2) = p(1);
       point_matrix(i, 3) = p(0) * p(1);
     }

   // Columns of coeff_matrix are the coefficients of the polynomial
   coeff_matrix.invert(point_matrix);
 }


 template <>
 Q1Basis<3>::Q1Basis(
   const typename Triangulation<3>::active_cell_iterator &cell)
   : Function<3>()
   , index_basis(0)
   , coeff_matrix(8, 8)
 {
   FullMatrix<double> point_matrix(8, 8);

   for (unsigned int i = 0; i < 8; ++i)
     {
       const Point<3> &p = cell->vertex(i);

       point_matrix(i, 0) = 1;
       point_matrix(i, 1) = p(0);
       point_matrix(i, 2) = p(1);
       point_matrix(i, 3) = p(2);
       point_matrix(i, 4) = p(0) * p(1);
       point_matrix(i, 5) = p(1) * p(2);
       point_matrix(i, 6) = p(0) * p(2);
       point_matrix(i, 7) = p(0) * p(1) * p(2);
     }

   // Columns of coeff_matrix are the coefficients of the polynomial
   coeff_matrix.invert(point_matrix);
 }


 template <int dim>
 void
 Q1Basis<dim>::set_index(unsigned int index)
 {
   index_basis = index;
 }


 template <>
 double
 Q1Basis<2>::value(const Point<2> &p, const unsigned int /* component */) const
 {
   double value = coeff_matrix(0, index_basis) +
                  coeff_matrix(1, index_basis) * p(0) +
                  coeff_matrix(2, index_basis) * p(1) +
                  coeff_matrix(3, index_basis) * p(0) * p(1);

   return value;
 }


 template <>
 double
 Q1Basis<3>::value(const Point<3> &p, const unsigned int /* component */) const
 {
   double value = coeff_matrix(0, index_basis) +
                  coeff_matrix(1, index_basis) * p(0) +
                  coeff_matrix(2, index_basis) * p(1) +
                  coeff_matrix(3, index_basis) * p(2) +
                  coeff_matrix(4, index_basis) * p(0) * p(1) +
                  coeff_matrix(5, index_basis) * p(1) * p(2) +
                  coeff_matrix(6, index_basis) * p(0) * p(2) +
                  coeff_matrix(7, index_basis) * p(0) * p(1) * p(2);

   return value;
 }


 template <>
 void
 Q1Basis<2>::value_list(const std::vector<Point<2>> &points,
                        std::vector<double> &        values,
                        const unsigned int /*component = 0*/) const
 {
   Assert(points.size() == values.size(),
          ExcDimensionMismatch(points.size(), values.size()));

   for (unsigned int p = 0; p < points.size(); ++p)
     {
       values[p] = coeff_matrix(0, index_basis) +
                   coeff_matrix(1, index_basis) * points[p](0) +
                   coeff_matrix(2, index_basis) * points[p](1) +
                   coeff_matrix(3, index_basis) * points[p](0) * points[p](1);

     } // end ++p
 }


 template <>
 void
 Q1Basis<3>::value_list(const std::vector<Point<3>> &points,
                        std::vector<double> &        values,
                        const unsigned int /*component = 0*/) const
 {
   Assert(points.size() == values.size(),
          ExcDimensionMismatch(points.size(), values.size()));

   for (unsigned int p = 0; p < points.size(); ++p)
     {
       values[p] = coeff_matrix(0, index_basis) +
                   coeff_matrix(1, index_basis) * points[p](0) +
                   coeff_matrix(2, index_basis) * points[p](1) +
                   coeff_matrix(3, index_basis) * points[p](2) +
                   coeff_matrix(4, index_basis) * points[p](0) * points[p](1) +
                   coeff_matrix(5, index_basis) * points[p](1) * points[p](2) +
                   coeff_matrix(6, index_basis) * points[p](0) * points[p](2) +
                   coeff_matrix(7, index_basis) * points[p](0) * points[p](1) *
                     points[p](2);

     } // end ++p
 }

} // end namespace Coefficients



#endif /* DIFFUSION_PROBLEM_INCLUDE_Q1BASIS_HPP_ */
