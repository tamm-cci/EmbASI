module EmbASI_qm_embedding_interface
  !
  ! This module provides a set of template functions to streamline
  ! the implementation of the EmbASI interface within a given software
  ! packages. They also provide some guidance on where they should
  ! be placed in the SCF cycle.
  !
  ! On the whole, these functions are wrappers around the pre-existing
  ! ASI callback routines - please refer to the implementation
  ! documentation provided for the ASI Callbacks for a given QM code.
  !
  ! This module covers four tasks:
  ! - The import/export of the density matrix
  ! - The export of the full Hamiltonian
  ! - The export of the two-electron Hamiltonian
  ! - The export of the one-electron Hamiltonian
  ! - The construction of the embedding potential
  !
  ! All of these routines may not be necessary. For example, the import
  ! of the Hamiltonian/density matrix may be handled by the ASI
  ! implementation elsewhere, but they are provided for completion
  ! sake. It may also be helpful to separate out the infrastructure
  ! for running embedding calculations with another set of keywords.
  !
  ! The functions provided are implementation dependent - they require
  ! some knowledge of where certain quantities are calculated and
  ! where they are defined.
  !
  implicit none

  type :: EmbASI_qm_embedding

     ! Parameters for control flow
     ! Full self-consistent Kohn-Sham calculation
     integer, parameter :: EMBASI_SC_KS_CALC = 1
     ! Non-self-consistent energy calculation for reference
     integer, parameter :: EMBASI_NONSC_CALC = 2
     ! Embedded self-consistent calculation
     integer, parameter :: EMBASI_SC_EMBEDDED_CALC = 3

   contains

     procedure :: export_overlap => export_EmbASI_overlap

     procedure :: export_densmat => export_EmbASI_densmat
     procedure :: import_densmat => import_EmbASI_densmat

     procedure :: export_H1e => export_EmbASI_H1e
     procedure :: export_H2e => export_EmbASI_H2e
     procedure :: export_Htot => export_EmbASI_Htot
     procedure :: export_allH => export_EmbASI_allH

     procedure :: set_embedding_H => set_EmbASI_embedding_H

  end type EmbASI_qm_embedding

  type(EmbASI_qm_embedding) :: QM_Embedding_QMCode

contains

  subroutine export_EmbASI_overlap(this, iS, iK, overlap, blacs_descr)
    !> Wrapper for export of the overlap matrix with ASI
    use asi_callbacks, only : MyCode_ASI_Callbacks

    class(EmbASI_qm_embedding) :: this

    integer, intent(in) :: iS
    integer, intent(in) :: iK

    real(8), intent(out) :: overlap(:,:)

    integer, intent(in), optional :: blacs_descr(:)

    if (present(blacs_descr)) then
       call MyCode_ASI_Callbacks%invoke_overlap(iS, iK, overlap, blacs_descr)
    else
       call MyCode_ASI_Callbacks%invoke_overlap(iS, iK, overlap)
    end if

  end subroutine export_EmbASI_overlap


  subroutine export_EmbASI_densmat(this, iS, iK, outdensmat,&
       & blacs_descr)
    !> Wrapper for ASI export function of the density matrix
    use asi_callbacks, only : MyCode_ASI_Callbacks

    class(EmbASI_qm_embedding) :: this

    integer, intent(in) :: iS
    integer, intent(in) :: iK

    real(8), intent(out) :: output_densmat(:,:)

    integer, intent(in), optional :: blacs_descr(:)

    if (present(blacs_descr)) then
       call MyCode_ASI_Callbacks%invoke_dm(iS, iK, output_densmat, blacs_descr)
    else
       call MyCode_ASI_Callbacks%invoke_dm(iS, iK, output_densmat)
    end if

  end subroutine export_EmbASI_densmat


  subroutine import_EmbASI_densmat(this, iS, iK, outdensmat, blacs_descr)
    !> Wrapper for ASI import function of the density matrix
    use asi_callbacks, only : MyCode_ASI_Callbacks

    class(EmbASI_qm_embedding) :: this

    integer, intent(in) :: iS
    integer, intent(in) :: iK

    real(8), intent(out) :: input_densmat(:,:)

    integer, intent(in), optional :: blacs_descr(:)

    if (present(blacs_descr)) then
       call MyCode_ASI_Callbacks%invoke_dm_init(iS, iK, input_densmat, blacs_descr)
    else
       call MyCode_ASI_Callbacks%invoke_dm_init(iS, iK, input_densmat, H1e)
    end if

  end subroutine import_EmbASI_densmat


  subroutine export_EmbASI_H1e(this, iS, iK, H1e, blacs_descr)
    !> Wrapper for ASI export function of the one-electron Hamiltonian
    use asi_callbacks, only : MyCode_ASI_Callbacks

    class(EmbASI_qm_embedding) :: this

    real(8), intent(in) :: H1e(:,:)

    integer, intent(in) :: iS
    integer, intent(in) :: iK

    integer, intent(in), optional :: blacs_descr(:)

    if (present(blacs_descr)) then
       call MyCode_ASI_Callbacks%invokeH(iS, iK, H1e, blacs_descr)
    else
       call MyCode_ASI_Callbacks%invokeH(iS, iK, H1e)
    end if

  end subroutine export_EmbASI_H1e


  subroutine export_EmbASI_H2e(this, iS, iK, H2e, blacs_descr)
    !> Wrapper for ASI export function of the two-electron
    !  Hamiltonian components
    use asi_callbacks, only : MyCode_ASI_Callbacks

    class(EmbASI_qm_embedding) :: this

    real(8), intent(in) :: H2e(:,:)

    integer, intent(in) :: iS
    integer, intent(in) :: iK

    integer, intent(in), optional :: blacs_descr(:)

    if (present(blacs_descr)) then
       call MyCode_ASI_Callbacks%invokeH(iS, iK, H2e, blacs_descr)
    else
       call MyCode_ASI_Callbacks%invokeH(iS, iK, H2e)
    end if

  end subroutine export_EmbASI_H2e


  subroutine export_EmbASI_Htot(this, iS, iK, Htot, blacs_descr)
    !> Wrapper for ASI export function of the total Hamiltonian.
    use asi_callbacks, only : MyCode_ASI_Callbacks

    class(EmbASI_qm_embedding) :: this

    real(8), intent(in) :: Htot(:,:)

    integer, intent(in) :: iS
    integer, intent(in) :: iK

    integer, intent(in), optional :: blacs_descr(:)

    if (present(blacs_descr)) then
       call MyCode_ASI_Callbacks%invokeH(iS, iK, Htot, blacs_descr)
    else
       call MyCode_ASI_Callbacks%invokeH(iS, iK, Htot)
    end if

  end subroutine export_EmbASI_Htot

  subroutine export_EmbASI_all_H(this, iS, iK, H1e, H2e, Htot, blacs_descr)
    !> Wrapper for ASI export of the one-electron and two-electron
    !  components of the Hamiltonian and the total Hamiltonian
    class(EmbASI_qm_embedding) :: this

    real(8), intent(in) :: H1e(:,:)
    real(8), intent(in) :: H2e(:,:)
    real(8), intent(in) :: Htot(:,:)

    integer, intent(in) :: iS
    integer, intent(in) :: iK

    integer, intent(in), optional :: blacs_descr(:)

    if (present(blacs_descr)) then
       call this%export_EmbASI_H1e(iS, iK, H1e, blacs_descr)
       call this%export_EmbASI_H2e(iS, iK, H2e, blacs_descr)
       call this%export_EmbASI_Htot(iS, iK, Htot, blacs_descr)
    else
       call this%export_EmbASI_H1e(iS, iK, H1e)
       call this%export_EmbASI_H2e(iS, iK, H2e)
       call this%export_EmbASI_Htot(iS, iK, Htot)
    end if

  end subroutine export_EmbASI_all_H


  subroutine set_EmbASI_embedding_H(this, iS, iK,&
       & embedding_hamiltonian, blacs_descr)
    !> Wrapper for the construction of the embedding Hamiltonian
    use asi_callbacks, only : MyCode_ASI_Callbacks
    class(EmbASI_qm_embedding) :: this

    integer, intent(in) :: iS
    integer, intent(in) :: iK

    integer, intent(inout) :: embedding_hamiltonian
    integer, intent(in), optional :: blacs_descr(:)

    ! The export and modify pattern is part of a nasty cludge
    ! required for the self-consistent Huzinaga implementation.
    ! Parts of the Hamiltonian constructed in the SCF cycle are
    ! needed to build the projection embedding function, so those
    ! parts are constructed in the callback function and then added
    ! to the embedded Hamiltonian.
    !
    ! Sorry for the cludge.
    if (present(blacs_descr)) then
       call this%export_EmbASI_Htot(iS, iK, embedding_hamiltonian,&
            & blacs_descr)
       call MyCode_ASI_Callbacks%modify_hamiltonian(iS, iK,&
            & embedding_hamiltonian, blacs_descr)
    else
       call this%export_EmbASI_Htot(iS, iK, embedding_hamiltonian)
       call MyCode_ASI_Callbacks%modify_hamiltonian(iS, iK, embedding_hamiltonian)
    end if

  end subroutine set_EmbASI_set_embedding_H


end module EmbASI_qm_embedding_interface
