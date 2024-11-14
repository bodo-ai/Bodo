package com.bodosql.calcite.rel.metadata

import org.apache.calcite.config.CalciteSystemProperty
import org.apache.calcite.interpreter.JaninoRexCompiler
import org.apache.calcite.rel.metadata.JaninoRelMetadataProvider
import org.apache.calcite.rel.metadata.MetadataHandler
import org.apache.calcite.rel.metadata.MetadataHandlerProvider
import org.apache.calcite.rel.metadata.RelMetadataProvider
import org.apache.calcite.rel.metadata.janino.RelMetadataHandlerGeneratorUtil
import org.codehaus.commons.compiler.CompileException
import org.codehaus.commons.compiler.CompilerFactoryFactory
import org.codehaus.commons.compiler.ICompilerFactory
import org.codehaus.commons.compiler.ISimpleCompiler
import java.io.IOException
import java.lang.IllegalArgumentException
import java.lang.reflect.Constructor
import java.lang.reflect.InvocationTargetException
import java.util.Objects
import java.util.stream.Collectors

/**
 * Implementation of MetadataHandlerProvider to enable updating cluster.setMetadataQuerySupplier.
 * This is largely based on JaninoRelMetadataProvider inside Calcite.
 */
class BodoRelMetadataHandlerProvider(
    private val relMetadataProvider: RelMetadataProvider,
) : MetadataHandlerProvider {
    /**
     * This is largely replicated from both Calcite and Dremio. My understanding of the code (which
     * may not be fully correct) is:
     *
     * 1. Fetch the set of Handler implementations for a given Class. I believe this handles chaining.
     *
     * 2. Generate a class name and source code for the handlers. I believe
     * we are basically JIT compiling this class.
     *
     * 3. The "compile" function below basically generates the code to be
     * compiled into usable Java classes for each handler's implementation.
     * This is done with the Janino package.
     *
     */
    override fun <MH : MetadataHandler<*>?> handler(handlerClass: Class<MH>): MH {
        val handlers =
            relMetadataProvider
                .handlers(handlerClass)
                .stream()
                .distinct()
                .collect(Collectors.toList())
        val generatedHandler = RelMetadataHandlerGeneratorUtil.generateHandler(handlerClass, handlers)
        return compile(generatedHandler.handlerName, generatedHandler.generatedCode, handlerClass, handlers)
    }

    @Throws(CompileException::class, IOException::class)
    private fun <MH : MetadataHandler<*>?> compile(
        className: String?,
        generatedCode: String?,
        handlerClass: Class<MH>,
        argList: List<Any?>,
    ): MH {
        val compilerFactory: ICompilerFactory
        val classLoader =
            Objects.requireNonNull(
                JaninoRelMetadataProvider::class.java.classLoader,
                "classLoader",
            )
        compilerFactory =
            try {
                CompilerFactoryFactory.getDefaultCompilerFactory(classLoader)
            } catch (e: Exception) {
                throw IllegalStateException(
                    "Unable to instantiate java compiler",
                    e,
                )
            }
        val compiler: ISimpleCompiler = compilerFactory.newSimpleCompiler()
        compiler.setParentClassLoader(JaninoRexCompiler::class.java.classLoader)
        if (CalciteSystemProperty.DEBUG.value()) {
            // Add line numbers to the generated janino class
            compiler.setDebuggingInformation(true, true, true)
            println(generatedCode)
        }
        compiler.cook(generatedCode)
        val constructor: Constructor<*>
        val o: Any
        try {
            constructor =
                compiler.classLoader
                    .loadClass(className)
                    .declaredConstructors[0]
            o = constructor.newInstance(*argList.toTypedArray())
        } catch (e: IllegalArgumentException) {
            throw java.lang.RuntimeException(e)
        } catch (e: InstantiationException) {
            throw RuntimeException(e)
        } catch (e: IllegalAccessException) {
            throw RuntimeException(e)
        } catch (e: InvocationTargetException) {
            throw RuntimeException(e)
        } catch (e: ClassNotFoundException) {
            throw RuntimeException(e)
        }
        return handlerClass.cast(o)
    }
}
