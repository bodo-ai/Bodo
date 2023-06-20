package com.bodosql.calcite.prepare

import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import org.apache.calcite.plan.ConventionTraitDef
import org.apache.calcite.plan.RelTrait
import org.apache.calcite.plan.RelTraitDef

enum class PlannerType {
    HEURISTIC {
        override fun programs(): ProgramCollection =
            ProgramCollection(
                BodoPrograms.subQueryRemove(),
                // We create a new program each time we construct a new planner.
                // This is because calcite 1.30.0's hep program is not threadsafe
                // so we're just going to be careful and not use a singleton.
                //
                // Version 1.32.0 and beyond has fixed this and it's no longer
                // necessary there.
                BodoPrograms.hepStandard(optimize = false),
                BodoPrograms.hepStandard(optimize = true)
            )
    },

    VOLCANO {
        override fun programs(): ProgramCollection =
            ProgramCollection(
                BodoPrograms.subQueryRemove(),
                BodoPrograms.standard(optimize = false),
                BodoPrograms.standard(optimize = true),
            )
    },

    STREAMING {
        override fun traitDefs(): List<RelTraitDef<out RelTrait>> =
            listOf(ConventionTraitDef.INSTANCE, BatchingPropertyTraitDef.INSTANCE)

        override fun programs(): ProgramCollection = VOLCANO.programs()
    };

    open fun traitDefs(): List<RelTraitDef<out RelTrait>> =
        // Override the list of standard trait definitions we are interested in.
        // This determines what traits are physically possible for us to process.
        // We override this to include only the convention trait because we
        // haven't yet ensured the collation trait is properly supported
        // and will always yield possible plans.
        // We'll probably want to re-enable the collation trait at that time,
        // but there's no point in potentially failing a plan for a trait we
        // don't care about.
        listOf(ConventionTraitDef.INSTANCE)

    abstract fun programs(): ProgramCollection
}
