/*
 * diffusion_problem_ms.hpp
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

#ifndef DIFFUSION_PROBLEM_INCLUDE_DIFFUSION_PROBLEM_MS_HPP_
#define DIFFUSION_PROBLEM_INCLUDE_DIFFUSION_PROBLEM_MS_HPP_


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>


#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/base/convergence_table.h>
#include <fstream>
#include <iostream>

// My Headers
#include "diffusion_basis.hpp"
#include "dirichlet_bc.hpp"
#include "matrix_coeff.hpp"
#include "neumann_bc.hpp"
#include "right_hand_side.hpp"
#include "solution_gaussian.hpp"
#include "config.h"

/*!
 * @namespace DiffusionProblem
 * @brief Contains implementation of the main object
 * and all functions to solve a
 * Dirichlet-Neumann problem on a unit square.
 */
namespace DiffusionProblem
{
  using namespace dealii;

  /*!
   * @class DiffusionProblemMultiscale
   * @brief Main class to solve
   * Dirichlet-Neumann problem on a unit square with
   * multiscale FEM.
   */
  template <int dim>
  class DiffusionProblemMultiscale
  {
  public:
    /*!
     * Constructor.
     */
    DiffusionProblemMultiscale(unsigned int n_refine,
                               unsigned int n_refine_local);
    /*!
     * @brief Run function of the object.
     *
     * Run the computation after object is built.
     */
    void
    run();

  private:
    /*!
     * @brief Set up the grid with a certain number of refinements.
     *
     * Generate a triangulation of \f$[0,1]^{\rm{dim}}\f$ with edges/faces
     * numbered form \f$1,\dots,2\rm{dim}\f$.
     */
    void
    make_grid();

    /*!
     * Set all relevant data to local basis object and initialize the basis
     * fully. Then compute.
     */
    void
    initialize_and_compute_basis();

    /*!
     * @brief Setup sparsity pattern and system matrix.
     *
     * Compute sparsity pattern and reserve memory for the sparse system matrix
     * and a number of right-hand side vectors. Also build a constraint object
     * to take care of Dirichlet boundary conditions.
     */
    void
    setup_system();

    /*!
     * @brief Assemble the system matrix and the static right hand side.
     *
     * Assembly routine to build the time-independent (static) part.
     * Neumann boundary conditions will be put on edges/faces
     * with odd number. Constraints are not applied here yet.
     */
    void
    assemble_system();

    /*!
     * @brief Iterative solver.
     *
     * CG-based solver with AMG-preconditioning.
     */
    void
    solve_iterative();

    /*!
     * @brief Send coarse weights to corresponding local cell.
     *
     * After the coarse (global) weights have been computed they
     * must be set to the local basis object and stored there.
     * This is necessary to write the local multiscale solution.
     */
    void
    send_global_weights_to_cell();

    /*!
     * @brief Write coarse solution to disk.
     *
     * Write results for coarse solution to disk in vtu-format.
     */
    void
    output_result() const;

    /*!
     * Collect local file names on all mpi processes to write
     * the global pvtu-record.
     */
    std::vector<std::string>
    collect_filenames_on_mpi_process() const;

    void   compute_errors();

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    /*!
     *
     */
    LA::MPI::SparseMatrix system_matrix;

    /*!
     * Solution vector containing weights at the dofs.
     */
    LA::MPI::Vector solution;

    /*!
     * Contains all parts of the right-hand side needed to
     * solve the linear system.
     */
    LA::MPI::Vector system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    /*!
     * Number of global refinements.
     */
    const unsigned int n_refine;

    /*!
     * Number of local refinements.
     */
    const unsigned int n_refine_local;

    /*!
     * STL Vector holding basis functions for each coarse cell.
     */
    std::map<CellId, DiffusionProblemBasis<dim>> cell_basis_map;

    ConvergenceTable convergence_table;
  };
  
  template <int dim>
  DiffusionProblemMultiscale<dim>::DiffusionProblemMultiscale(
    unsigned int n_refine,
    unsigned int n_refine_local)
    :  mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening))
    , fe(1)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
    , n_refine(n_refine)
    , n_refine_local(n_refine_local)
    , cell_basis_map()
  {}


  template <int dim>
  void
  DiffusionProblemMultiscale<dim>::initialize_and_compute_basis()
  {
    TimerOutput::Scope t(computing_timer,
                         "basis initialization and computation");

    typename Triangulation<dim>::active_cell_iterator cell = dof_handler
                                                               .begin_active(),
                                                      endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            DiffusionProblemBasis<dim> current_cell_problem(
              n_refine_local,
              cell,
              triangulation.locally_owned_subdomain(),
              mpi_communicator);
            CellId current_cell_id(cell->id());

            std::pair<
              typename std::map<CellId, DiffusionProblemBasis<dim>>::iterator,
              bool>
              result;
            result = cell_basis_map.insert(
              std::make_pair(cell->id(), current_cell_problem));

            Assert(result.second,
                   ExcMessage(
                     "Insertion of local basis problem into std::map failed. "
                     "Problem with copy constructor?"));
          }
      } // end ++cell


    /*
     * Now each node possesses a set of basis objects.
     * We need to compute them on each node and do so in
     * a locally threaded way.
     */
    typename std::map<CellId, DiffusionProblemBasis<dim>>::iterator
      it_basis    = cell_basis_map.begin(),
      it_endbasis = cell_basis_map.end();
    for (; it_basis != it_endbasis; ++it_basis)
      {
        (it_basis->second).run();
      }

  }


  template <int dim>
  void
  DiffusionProblemMultiscale<dim>::make_grid()
  {
    TimerOutput::Scope t(computing_timer, "global mesh generation");

    GridGenerator::hyper_cube(triangulation, 0, 1, /* colorize */ true);

    triangulation.refine_global(n_refine);

    pcout << "Number of active global cells: " << triangulation.n_active_cells()
          << std::endl;

    std::cout <<  " MultiscaleFEM Coarse tri memory--- "
                     << triangulation.memory_consumption () << std::endl;
  }


  template <int dim>
  void
  DiffusionProblemMultiscale<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "global system setup");

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

   solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);

    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // Set up Dirichlet boundary conditions.
    const Coefficients::DirichletBC<dim> dirichlet_bc;
    for (unsigned int i = 0; i < dim; ++i)
      {
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 /*boundary id*/
                                                   i, // only even boundary id
                                                 dirichlet_bc,
                                                 constraints);
      }

    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs =*/true);
    SparsityTools::distribute_sparsity_pattern(
      dsp,
      dof_handler.locally_owned_dofs(),
      mpi_communicator,
      locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);

    std::cout <<  " MultiscaleFEM Coarse dof memory--- "
    		<< dof_handler.memory_consumption () << std::endl;
  }


  template <int dim>
  void
  DiffusionProblemMultiscale<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "global multiscale assembly");

    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    /*
     * Neumann BCs and vector to store the values.
     */
    const Coefficients::NeumannBC<dim> neumann_bc;
    std::vector<double>                neumann_values(n_face_q_points);

    /*
     * Integration over cells.
     */
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            typename std::map<CellId, DiffusionProblemBasis<dim>>::iterator
              it_basis = cell_basis_map.find(cell->id());

            cell_matrix = 0;
            cell_rhs    = 0;

            cell_matrix = (it_basis->second).get_global_element_matrix();
            cell_rhs    = (it_basis->second).get_global_element_rhs();

            /*
             * Boundary integral for Neumann values for odd boundary_id.
             */
            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number)
              {
                if (cell->face(face_number)->at_boundary() &&
                    ((cell->face(face_number)->boundary_id() == 11) ||
                     (cell->face(face_number)->boundary_id() == 13) ||
                     (cell->face(face_number)->boundary_id() == 15)))
                  {
                    fe_face_values.reinit(cell, face_number);

                    // Fill in values at this particular face.
                    neumann_bc.value_list(
                      fe_face_values.get_quadrature_points(), neumann_values);

                    for (unsigned int q_face_point = 0;
                         q_face_point < n_face_q_points;
                         ++q_face_point)
                      {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            cell_rhs(i) +=
                              neumann_values[q_face_point] // g(x_q)
                              * fe_face_values.shape_value(
                                  i, q_face_point)                // phi_i(x_q)
                              * fe_face_values.JxW(q_face_point); // dS
                          }                                       // end ++i
                      } // end ++q_face_point
                  }     // end if
              }         // end ++face_number


            // get global indices
            cell->get_dof_indices(local_dof_indices);
            /*
             * Now add the cell matrix and rhs to the right spots
             * in the global matrix and global rhs. Constraints will
             * be taken care of later.
             */
            constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
          }
      } // end ++cell

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }


  template <int dim>
  void
  DiffusionProblemMultiscale<dim>::solve_iterative()
  {
    TimerOutput::Scope t(computing_timer, "global iterative solver");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

#ifdef USE_PETSC_LA
    LA::SolverCG solver(solver_control, mpi_communicator);
#else
    LA::SolverCG solver(solver_control);
#endif

    LA::MPI::PreconditionAMG                 preconditioner;
    LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#else
    /* Trilinos defaults are good */
#endif

    preconditioner.initialize(system_matrix, data);

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);

    pcout << "   Global problem solved in " << solver_control.last_step()
          << " iterations." << std::endl;

    constraints.distribute(completely_distributed_solution);
   solution = completely_distributed_solution;
   pcout <<  "Multiscale FEM Coarse Solution MEMORY --- "<< solution.memory_consumption()
    	    	          << "  " << std::endl;
  }


  template <int dim>
  void
  DiffusionProblemMultiscale<dim>::send_global_weights_to_cell()
  {
    // For each cell we get dofs_per_cell values
    const unsigned int                   dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // active cell iterator
    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            std::vector<double> extracted_weights(dofs_per_cell, 0);
            solution.extract_subvector_to(local_dof_indices,
                                                           extracted_weights);

            typename std::map<CellId, DiffusionProblemBasis<dim>>::iterator
              it_basis = cell_basis_map.find(cell->id());
            (it_basis->second).set_global_weights(extracted_weights);
          }
      } // end ++cell
  }

  template <int dim>
  std::vector<std::string>
  DiffusionProblemMultiscale<dim>::collect_filenames_on_mpi_process() const
  {
    std::vector<std::string> filename_list;

    typename std::map<CellId, DiffusionProblemBasis<dim>>::const_iterator
      it_basis    = cell_basis_map.begin(),
      it_endbasis = cell_basis_map.end();

    for (; it_basis != it_endbasis; ++it_basis)
      {
        filename_list.push_back((it_basis->second).get_filename_global());
      }

    return filename_list;
  }

  template <int dim>
  void
  DiffusionProblemMultiscale<dim>::output_result() const
  {
	  // write local fine solution
	    typename std::map<CellId, DiffusionProblemBasis<dim>>::const_iterator it_basis = cell_basis_map
	                                                                        .begin(),
	                                                         it_endbasis =
	                                                             cell_basis_map.end();

	    for (; it_basis != it_endbasis; ++it_basis)
	    {
	      (it_basis->second).output_global_solution_in_cell();
	    }

	    // Gather local filenames
	    std::vector<std::string> filenames_on_cell;
	    {
	      std::vector<std::vector<std::string>> filename_list_list =
	          Utilities::MPI::gather(mpi_communicator,
	                                 collect_filenames_on_mpi_process(),
	                                 /* root_process = */ 0);

	      for (unsigned int i = 0; i < filename_list_list.size(); ++i)
	        for (unsigned int j = 0; j < filename_list_list[i].size(); ++j)
	          filenames_on_cell.emplace_back(filename_list_list[i][j]);
	    }

	    std::string filename = (dim == 2 ? "solution-ms_2d" : "solution-ms_3d");
	    DataOut<dim> data_out;
	    data_out.attach_dof_handler(dof_handler);
	    data_out.add_data_vector(solution, "u");

	    Vector<float> subdomain(triangulation.n_active_cells());
	    for (unsigned int i = 0; i < subdomain.size(); ++i)
	    {
	      subdomain(i) = triangulation.locally_owned_subdomain();
	    }
	    data_out.add_data_vector(subdomain, "subdomain");

	    // Postprocess
	    //      std::unique_ptr<Q_PostProcessor> postprocessor(
	    //        new Q_PostProcessor(parameter_filename));
	    //      data_out.add_data_vector(locally_relevant_solution, *postprocessor);

	    data_out.build_patches();

	    std::string filename_local_coarse(filename);
	    filename_local_coarse +=
	        "_coarse_refinements-" + Utilities::int_to_string(n_refine, 2) + "." +
	        Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
	        ".vtu";

	    std::ofstream output(filename_local_coarse.c_str());
	    data_out.write_vtu(output);

	    /*
	     * Write a pvtu-record to collect all files for each time step.
	     */
	    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
	    {
	      std::vector<std::string> all_local_filenames_coarse;
	      for (unsigned int i = 0;
	           i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
	      {
	        all_local_filenames_coarse.push_back(
	            filename + "_coarse_refinements-" +
	            Utilities::int_to_string(n_refine, 2) + "."  +
	            Utilities::int_to_string(i, 4) + ".vtu");
	      }

	      std::string filename_master(filename);

	      filename_master += "_coarse_refinements-" +
	                         Utilities::int_to_string(n_refine, 2) + "."  + ".pvtu";

	      std::ofstream master_output(filename_master);
	      data_out.write_pvtu_record(master_output, all_local_filenames_coarse);
	    }

	    // pvtu-record for all local fine outputs
	    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
	    {
	      std::string filename_master = filename;
	      filename_master += "_fine_refinements-" +
	                         Utilities::int_to_string(n_refine, 2) + "." + ".pvtu";

	      std::ofstream master_output(filename_master);
	      data_out.write_pvtu_record(master_output, filenames_on_cell);
	    }
  }


  template <int dim>
  void DiffusionProblemMultiscale<dim>::compute_errors()
  {

	    Vector<float> difference_per_cell(triangulation.n_active_cells());
	    VectorTools::integrate_difference(dof_handler,
	                                      solution,
										  Solution<dim>(),
	                                      difference_per_cell,
	                                     QGauss<dim>(fe.degree + 1),
	                                      VectorTools::L2_norm);

	    const double L2_error  = VectorTools::compute_global_error(triangulation,
	                                                 difference_per_cell,
	                                                 VectorTools::L2_norm);


	  const unsigned int n_active_cells = triangulation.n_active_cells();
	  const unsigned int n_dofs = dof_handler.n_dofs();
	    VectorTools::integrate_difference(dof_handler,
	                                     solution,
										 Solution<dim>(),
	                                      difference_per_cell,
	                                     QGauss<dim>(fe.degree + 1),
	                                      VectorTools::H1_norm);

	  const double  H1_error = VectorTools::compute_global_error(triangulation,
	                                                 difference_per_cell,
	                                                 VectorTools::H1_norm);

	  VectorTools::integrate_difference(dof_handler,
			                           solution,
									   Solution<dim>(),
	                                    difference_per_cell,
										QGauss<dim>(fe.degree + 1),
	                                    VectorTools::Linfty_norm);
	  const double Linfty_error =
	    VectorTools::compute_global_error(triangulation,
	                                      difference_per_cell,
	                                      VectorTools::Linfty_norm);
	    pcout << "   Number of active cells:       "
	         << n_active_cells
	         << std::endl
	         << "   Number of degrees of freedom: "
	         << n_dofs
	         << std::endl;
	   convergence_table.add_value("cells", n_active_cells);
	  convergence_table.add_value("dofs", n_dofs);
	  convergence_table.add_value("L2", L2_error);
	  convergence_table.add_value("H1", H1_error);
	  convergence_table.add_value("Linfty", Linfty_error);
	    convergence_table.set_precision("L2", 3);
	  convergence_table.set_precision("H1", 3);
	  convergence_table.set_precision("Linfty", 3);
			convergence_table.set_scientific("L2", true);
			convergence_table.set_scientific("H1", true);
			convergence_table.set_scientific("Linfty", true);
			convergence_table.set_tex_caption("cells", "\\# cells");
			convergence_table.set_tex_caption("dofs", "\\# dofs");
			convergence_table.set_tex_caption("L2", "L^2-error");
			convergence_table.set_tex_caption("H1", "H^1-error");
			convergence_table.set_tex_caption("Linfty", "L^\\infty-error");

			convergence_table.set_tex_format("cells", "r");
			convergence_table.set_tex_format("dofs", "r");

	    std::cout << std::endl;
	  convergence_table.write_text(std::cout);

	  std::ofstream error_table_file("tex-conv-table-msfem.tex");
	  convergence_table.write_tex(error_table_file);

	  deallog << "  Error in the L2 norm         : " << L2_error << std::endl;
	          deallog   << "  Error in the H1 norm     : " << H1_error <<std::endl;
	        		  deallog   << "  Error in the Linfty norm     : " << H1_error
	        		             << std::endl;
  }



  template <int dim>
  void
  DiffusionProblemMultiscale<dim>::run()
  {
    pcout << std::endl
          << "===========================================" << std::endl
          << "Solving >> MULTISCALE << problem in " << dim << "D." << std::endl;

    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    make_grid();

    setup_system();

    initialize_and_compute_basis();

    assemble_system();

    // Now solve
    solve_iterative();

    send_global_weights_to_cell();

    if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 90)
      {
        TimerOutput::Scope t(computing_timer, "coarse output vtu");
        output_result();
      }

    deallog << "Solve" << std::endl;
         compute_errors();
             convergence_table.set_precision("L2", 3);
       convergence_table.set_precision("H1", 3);

        convergence_table.set_scientific("L2", true);
       convergence_table.set_scientific("H1", true);
    computing_timer.print_summary();
    computing_timer.reset();

  Utilities::System::MemoryStats stats;
        Utilities::System::get_memory_stats(stats);
        pcout << "Peak virtual memory used, resident in kB: " << stats.VmSize
              << ' ' << stats.VmRSS << std::endl;

    pcout << std::endl
          << "===========================================" << std::endl;
  }

} // end namespace DiffusionProblem



#endif /* DIFFUSION_PROBLEM_INCLUDE_DIFFUSION_PROBLEM_MS_HPP_ */
